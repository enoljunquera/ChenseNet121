#!/usr/bin/env python3
"""
Leakage-safe NO-DOCKING baselines for LD50.

Models:
  - Ridge(alpha=1.0)
  - LinearSVR(C=1.0, epsilon=0.0, max_iter=50000)
  - RandomForestRegressor(n_estimators=500, min_samples_leaf=1)

Inputs used by the models (NO docking information):
  - MolWt
  - TPSA
  - MolLogP
  - NumRotatableBonds

For fair ablation against the full ChenseNet121 rerun, the eligible cohort is matched
using the same multimodal availability requirements (image + full float/bool voxel +
affinities present), but NONE of those docking/image variables is provided to these
models. This should reproduce the same 249 retained rows / 199 unique structures when
run on the same dataset used for ChenseNet121.

Revision safeguards:
  - exact molecular structure (InChIKey; canonical SMILES fallback) defines groups;
  - if any Code belonging to a structure is in an original test CSV, every retained row
    of that structure is assigned to that effective test set;
  - no exact structure may occur in both train and test;
  - ambiguous Code->multiple-structure rows and structures with conflicting oral LD50
    are excluded with an audit trail;
  - log1p target transform is preserved from the previous classical baselines;
  - predictions are back-transformed to mg/kg and clipped at zero;
  - MAE/MSE/RMSE and 2000 mg/kg post-hoc classification metrics are saved both by
    row and by unique structure.
"""

from pathlib import Path
import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, inchi
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

ID_COL = "Code"
SMILES_COL = "SMILES"
TARGET_COL = "Acute Oral LD50"
N_TESTS = 10
RANDOM_STATE = 42
THRESHOLD = 2000.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="./dataset")
    p.add_argument("--results-dir", default="./results/baseline_no_docking_grouped")
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args()


def norm_code(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s or s.lower() == "code":
        return None
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def structure_key(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        k = inchi.MolToInchiKey(mol)
        if k:
            return k
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def four_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return [np.nan] * 4
    return [
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumRotatableBonds(mol),
    ]


def available_codes_from_3d_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    codes = set()
    for block in content.strip().split("**************************"):
        lines = block.strip().splitlines()
        if not lines:
            continue
        cid = norm_code(lines[0])
        if cid is not None:
            codes.add(cid)
    return codes


def resolve_image_path(dataset_dir, raw_path, code=None):
    candidates = []
    if not pd.isna(raw_path):
        s = str(raw_path).strip().replace("\\", "/")
        if s.startswith("./"):
            s = s[2:]
        candidates += [dataset_dir / s, Path(str(raw_path))]
    if code:
        candidates.append(dataset_dir / "images" / f"CID_{code}.png")
    for p in candidates:
        if p.exists():
            return p
    return None


def build_matched_cohort(dataset_dir, audit_dir):
    combined_path = dataset_dir / "combined_dataset.csv"
    bool_path = dataset_dir / "dataset_descriptores_bool.txt"
    float_path = dataset_dir / "dataset_descriptores_float.txt"
    for p in (combined_path, bool_path, float_path):
        if not p.exists():
            raise FileNotFoundError(f"Falta fichero requerido: {p.resolve()}")

    audit_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(combined_path).copy()
    df["OriginalRow"] = np.arange(len(df))
    df[ID_COL] = df[ID_COL].map(norm_code)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df["StructureKey"] = df[SMILES_COL].map(structure_key)

    code_n_struct = (
        df.dropna(subset=[ID_COL, "StructureKey"])
          .groupby(ID_COL)["StructureKey"].nunique()
    )
    ambiguous_codes = set(code_n_struct[code_n_struct > 1].index)

    conflict_keys = set()
    for key, grp in df.dropna(subset=["StructureKey"]).groupby("StructureKey"):
        if grp[TARGET_COL].dropna().nunique() > 1:
            conflict_keys.add(key)

    float_codes = available_codes_from_3d_file(float_path)
    bool_codes = available_codes_from_3d_file(bool_path)

    kept, excluded = [], []
    for _, row in df.iterrows():
        reasons = []
        cid = row[ID_COL]
        skey = row["StructureKey"]

        if cid is None:
            reasons.append("missing_or_invalid_code")
        if skey is None:
            reasons.append("invalid_smiles")
        if pd.isna(row[TARGET_COL]):
            reasons.append("missing_oral_ld50")
        if cid in ambiguous_codes:
            reasons.append("code_maps_to_multiple_structures")
        if skey in conflict_keys:
            reasons.append("structure_has_conflicting_oral_ld50")

        # Cohort matching only: these are NOT input features of the no-docking models.
        if cid not in float_codes:
            reasons.append("missing_full_float_voxel")
        if cid not in bool_codes:
            reasons.append("missing_full_bool_voxel")
        if resolve_image_path(dataset_dir, row.get("Image Path"), cid) is None:
            reasons.append("missing_image")
        affinities = pd.to_numeric(
            pd.Series([row.get("Affinity 7E3D"), row.get("Affinity 8DT2")]),
            errors="coerce",
        ).to_numpy(dtype=float)
        if np.isnan(affinities).any():
            reasons.append("missing_affinity_for_cohort_matching")

        if reasons:
            rr = row.to_dict()
            rr["ExclusionReason"] = ";".join(sorted(set(reasons)))
            excluded.append(rr)
        else:
            kept.append(row.to_dict())

    meta = pd.DataFrame(kept).reset_index(drop=True)
    pd.DataFrame(excluded).to_csv(audit_dir / "excluded_rows_all_reasons.csv", index=False)
    meta.to_csv(audit_dir / "retained_rows_metadata.csv", index=False)

    desc = np.asarray([four_rdkit_descriptors(s) for s in meta[SMILES_COL]], dtype=float)
    X = pd.DataFrame(
        desc,
        columns=["MolWt", "TPSA", "MolLogP", "NumRotatableBonds"],
        index=meta.index,
    )
    y = meta[TARGET_COL].astype(float)

    print(f"Filas originales: {len(df)}")
    print(f"Filas retenidas para comparación justa: {len(meta)}")
    print(f"Estructuras únicas retenidas: {meta['StructureKey'].nunique()}")
    print(f"Códigos ambiguos excluidos: {len(ambiguous_codes)}")
    print(f"Estructuras con target conflictivo excluidas: {len(conflict_keys)}")
    print(f"Features usadas por el modelo: {list(X.columns)}")
    print("Docking usado como input: NO")

    return meta, X, y


def log_target(regressor):
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def make_models():
    return {
        "Ridge_NoDock": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", log_target(Ridge(alpha=1.0))),
        ]),
        "LinearSVR_NoDock": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", log_target(LinearSVR(
                C=1.0,
                epsilon=0.0,
                max_iter=50000,
                random_state=RANDOM_STATE,
            ))),
        ]),
        "RandomForest_NoDock": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", log_target(RandomForestRegressor(
                n_estimators=500,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                min_samples_leaf=1,
            ))),
        ]),
    }


def class_metrics(y_true, y_pred, threshold=THRESHOLD):
    true_toxic = np.asarray(y_true) <= threshold
    pred_toxic = np.asarray(y_pred) <= threshold
    tp = int(np.sum(true_toxic & pred_toxic))
    fn = int(np.sum(true_toxic & ~pred_toxic))
    fp = int(np.sum(~true_toxic & pred_toxic))
    tn = int(np.sum(~true_toxic & ~pred_toxic))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall)
        else float("nan")
    )
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "precision": precision, "recall": recall, "F1": f1, "accuracy": accuracy}


def summarize(meta_test, y_true, y_pred):
    row_df = meta_test.reset_index(drop=True).copy()
    row_df["LD50_real"] = np.asarray(y_true, dtype=float)
    row_df["LD50_predicho"] = np.asarray(y_pred, dtype=float)
    row_df["AbsoluteError"] = np.abs(row_df["LD50_real"] - row_df["LD50_predicho"])

    struct_df = (
        row_df.groupby("StructureKey", as_index=False)
        .agg(
            Common_Name=("Common Name", lambda s: ";".join(sorted(set(map(str, s.dropna()))))),
            LD50_real=("LD50_real", "mean"),
            LD50_predicho=("LD50_predicho", "mean"),
            n_rows=("Code", "size"),
        )
    )
    struct_df["AbsoluteError"] = np.abs(struct_df["LD50_real"] - struct_df["LD50_predicho"])

    row_mse = mean_squared_error(row_df["LD50_real"], row_df["LD50_predicho"])
    struct_mse = mean_squared_error(struct_df["LD50_real"], struct_df["LD50_predicho"])
    m = {
        "row_MAE": float(mean_absolute_error(row_df["LD50_real"], row_df["LD50_predicho"])),
        "row_MSE": float(row_mse),
        "row_RMSE": float(np.sqrt(row_mse)),
        "structure_MAE": float(mean_absolute_error(struct_df["LD50_real"], struct_df["LD50_predicho"])),
        "structure_MSE": float(struct_mse),
        "structure_RMSE": float(np.sqrt(struct_mse)),
    }
    m.update({f"row_class_{k}": v for k, v in class_metrics(row_df["LD50_real"], row_df["LD50_predicho"]).items()})
    m.update({f"structure_class_{k}": v for k, v in class_metrics(struct_df["LD50_real"], struct_df["LD50_predicho"]).items()})
    return row_df, struct_df, m


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    results_root = Path(args.results_dir)
    audit_dir = results_root / "audit"
    results_root.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    meta, X_all, y_all = build_matched_cohort(dataset_dir, audit_dir)
    models = make_models()

    split_summary = []
    split_cache = {}
    for i in range(1, N_TESTS + 1):
        test_path = dataset_dir / f"test_{i}_updated.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Falta {test_path.resolve()}")
        raw = pd.read_csv(test_path)
        original_codes = {norm_code(v) for v in raw[ID_COL]}
        original_codes.discard(None)

        seed_groups = set(meta.loc[meta[ID_COL].isin(original_codes), "StructureKey"])
        if not seed_groups:
            raise RuntimeError(f"TEST {i}: ningún Code original sobrevivió al filtrado")

        te = np.flatnonzero(meta["StructureKey"].isin(seed_groups).to_numpy())
        test_groups = set(meta.iloc[te]["StructureKey"])
        tr = np.flatnonzero((~meta["StructureKey"].isin(test_groups)).to_numpy())
        train_groups = set(meta.iloc[tr]["StructureKey"])
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(f"TEST {i}: leakage train-test: {len(overlap)} estructuras")

        split_summary.append({
            "test": i,
            "train_rows": len(tr),
            "test_rows": len(te),
            "train_unique_structures": len(train_groups),
            "test_unique_structures": len(test_groups),
            "train_test_structure_overlap": len(overlap),
        })
        split_cache[i] = (tr, te, original_codes)

    split_df = pd.DataFrame(split_summary)
    split_df.to_csv(audit_dir / "split_summary.csv", index=False)
    print("\n=== AUDITORÍA DE SPLITS ===")
    print(split_df.to_string(index=False))
    print(f"Máximo solapamiento estructural train-test: {split_df['train_test_structure_overlap'].max()}")

    if len(meta) == 249 and meta["StructureKey"].nunique() == 199:
        print("\nOK: misma cohorte global que ChenseNet121 (249 filas / 199 estructuras).")
    else:
        print("\nAVISO: la cohorte no es 249 filas / 199 estructuras; revisa dataset.")

    if args.audit_only:
        print("\nAuditoría terminada. No se han entrenado modelos.")
        return

    all_metrics, all_preds = [], []
    for i in range(1, N_TESTS + 1):
        tr, te, original_codes = split_cache[i]
        X_train, y_train = X_all.iloc[tr], y_all.iloc[tr]
        X_test, y_test = X_all.iloc[te], y_all.iloc[te]
        meta_test = meta.iloc[te].copy()
        meta_test["WasListedByOriginalCode"] = meta_test[ID_COL].isin(original_codes)

        exp_dir = results_root / f"test_{i:02d}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        meta_test.to_csv(exp_dir / "effective_test_rows.csv", index=False)

        print("\n" + "="*80)
        print(f"TEST {i}: train rows={len(tr)}, test rows={len(te)}, overlap=0")
        print("="*80)

        for name, model in models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            pred = np.asarray(fitted.predict(X_test), dtype=float)
            pred = np.clip(pred, 0, None)

            row_df, struct_df, m = summarize(meta_test, y_test.to_numpy(dtype=float), pred)
            m.update({
                "test": i,
                "model": name,
                "n_features": 4,
                "uses_docking_input": False,
                "train_rows": len(tr),
                "test_rows": len(te),
                "train_unique_structures": int(meta.iloc[tr]["StructureKey"].nunique()),
                "test_unique_structures": int(meta.iloc[te]["StructureKey"].nunique()),
                "train_test_structure_overlap": 0,
            })

            row_df.insert(0, "model", name)
            row_df.insert(0, "test", i)
            struct_df.insert(0, "model", name)
            struct_df.insert(0, "test", i)
            row_df.to_csv(exp_dir / f"{name}_predictions_row_level.csv", index=False)
            struct_df.to_csv(exp_dir / f"{name}_predictions_structure_level.csv", index=False)
            with open(exp_dir / f"{name}_metrics.json", "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2, allow_nan=True)

            all_metrics.append(m)
            all_preds.append(row_df)
            print(f"  {name:20s} | MAE row={m['row_MAE']:.3f} | MSE row={m['row_MSE']:.3f} | MAE struct={m['structure_MAE']:.3f}")

    metrics_df = pd.DataFrame(all_metrics).sort_values(["model", "test"])
    metrics_df.to_csv(results_root / "metrics_by_test.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(results_root / "predictions_all_tests_row_level.csv", index=False)

    rows = []
    for name, g in metrics_df.groupby("model"):
        rows.append({
            "model": name,
            "n_tests": len(g),
            "row_MAE_mean": g["row_MAE"].mean(),
            "row_MAE_std": g["row_MAE"].std(ddof=1),
            "row_MSE_mean": g["row_MSE"].mean(),
            "row_MSE_std": g["row_MSE"].std(ddof=1),
            "structure_MAE_mean": g["structure_MAE"].mean(),
            "structure_MAE_std": g["structure_MAE"].std(ddof=1),
            "structure_MSE_mean": g["structure_MSE"].mean(),
            "structure_MSE_std": g["structure_MSE"].std(ddof=1),
            "row_precision_mean": g["row_class_precision"].mean(),
            "row_recall_mean": g["row_class_recall"].mean(),
            "row_F1_mean": g["row_class_F1"].mean(),
            "row_accuracy_mean": g["row_class_accuracy"].mean(),
            "max_train_test_structure_overlap": g["train_test_structure_overlap"].max(),
        })

    summary = pd.DataFrame(rows).sort_values("row_MAE_mean")
    summary.to_csv(results_root / "metrics_summary_by_model.csv", index=False)
    print("\n=== RESUMEN FINAL - SIN DOCKING ===")
    print(summary.to_string(index=False))
    print(f"\nResultados: {results_root.resolve()}")


if __name__ == "__main__":
    main()
