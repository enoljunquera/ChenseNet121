#!/usr/bin/env python3
"""Leakage-safe retraining of the ORIGINAL full ChenseNet121 LD50 model.

This script preserves the active modalities and architecture of the original
25x25x25 + SMILES descriptors + docking affinities + 2D images experiment:
  - 2D RGB molecular image branch
  - full 25x25x25x8 docking voxel branch (3D convolutions)
  - 6 tabular descriptors (4 RDKit descriptors + 2 affinities)

Corrections made for the revision:
  1. Inputs are aligned explicitly by Code (never by row/index position).
  2. Exact molecular structure (InChIKey, canonical SMILES fallback) is the split group.
  3. If any Code of a structure belongs to an original test CSV, all retained rows
     of that structure are assigned to the effective test set.
  4. GroupKFold prevents identical structures crossing train/validation.
  5. Ambiguous Code->multiple-structure rows and conflicting oral-LD50 structures
     are excluded with an audit trail. FlatDock-specific outlier filtering is NOT applied
     to the full ChenseNet121 experiment.
  6. The original 1000-epoch/Adam/MSE/ReduceLROnPlateau protocol is preserved by default.
  7. Both best-validation and final-epoch checkpoints are saved/evaluated. The primary
     reported metrics use best-validation checkpoints, matching the manuscript wording.

Expected project structure:
  densenet121.py
  hyperparams.txt
  trainChenseNet121_full_grouped.py
  dataset/
    combined_dataset.csv
    dataset_descriptores_bool.txt
    dataset_descriptores_float.txt
    images/
    test_1_updated.csv ... test_10_updated.csv

Usage:
  python trainChenseNet121_full_grouped.py --audit-only
  python trainChenseNet121_full_grouped.py
  python trainChenseNet121_full_grouped.py --test 1
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, inchi
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset, Subset

from densenet121 import FlexibleDenseNet

ID_COL = "Code"
SMILES_COL = "SMILES"
TARGET_COL = "Acute Oral LD50"
RANDOM_STATE = 42
N_TESTS = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="./dataset")
    p.add_argument("--results-dir", default="./results/chensenet121_full_grouped")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--test", type=int, default=None, help="Run only one test (1..10)")
    return p.parse_args()


def load_hyperparams(path="hyperparams.txt"):
    hp = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            k, v = raw.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                hp[k] = float(v)
            except ValueError:
                hp[k] = v
    return hp


def norm_code(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s or s.lower() == "code":
        return None
    # Avoid pandas' occasional numeric formatting, e.g. 123.0
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
        key = inchi.MolToInchiKey(mol)
        if key:
            return key
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def smiles_descriptors(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return np.asarray([
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumRotatableBonds(mol),
    ], dtype=np.float32)


def load_3d_descriptor_file(path):
    """Load 25x25x25x4 descriptor blocks keyed by Code."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    result = {}
    for block in content.strip().split("**************************"):
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        cid = norm_code(lines[0])
        if cid is None:
            continue
        numeric = " ".join(lines[1:]).replace("[", " ").replace("]", " ").replace("\t", " ")
        toks = [t for t in re.sub(r"\s+", " ", numeric).split(" ") if t]
        try:
            vals = np.asarray(list(map(float, toks)), dtype=np.float32)
        except ValueError as exc:
            raise ValueError(f"Valores no numéricos en {path}, Code={cid}") from exc
        if vals.size != 62500:
            raise ValueError(f"Se esperaban 62500 valores en {path}, Code={cid}; hay {vals.size}")
        result[cid] = vals.reshape(25, 25, 25, 4)
    return result


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
    return candidates[0] if candidates else None


def load_image(dataset_dir, raw_path, code, target_size=(300, 300)):
    p = resolve_image_path(dataset_dir, raw_path, code)
    if p is None or not p.exists():
        raise FileNotFoundError(f"Falta imagen para Code={code}: {p}")
    try:
        resample = Image.Resampling.NEAREST
    except AttributeError:
        resample = Image.NEAREST
    with Image.open(p) as im:
        im = im.convert("RGB").resize(target_size, resample=resample)
        return np.asarray(im, dtype=np.float32) / 255.0


def build_dataset(dataset_dir, audit_dir):
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

    invalid = df[df["StructureKey"].isna()].copy()
    invalid.to_csv(audit_dir / "excluded_invalid_smiles.csv", index=False)

    code_n_struct = df.dropna(subset=[ID_COL, "StructureKey"]).groupby(ID_COL)["StructureKey"].nunique()
    ambiguous_codes = set(code_n_struct[code_n_struct > 1].index)
    df[df[ID_COL].isin(ambiguous_codes)].to_csv(audit_dir / "ambiguous_codes_multiple_structures.csv", index=False)

    conflict_keys = set()
    for key, grp in df.dropna(subset=["StructureKey"]).groupby("StructureKey"):
        if grp[TARGET_COL].dropna().nunique() > 1:
            conflict_keys.add(key)
    df[df["StructureKey"].isin(conflict_keys)].to_csv(audit_dir / "conflicting_oral_ld50_structures.csv", index=False)

    print("Cargando volúmenes 25x25x25x4 float...")
    vol_float = load_3d_descriptor_file(float_path)
    print(f"  {len(vol_float)} códigos")
    print("Cargando volúmenes 25x25x25x4 bool...")
    vol_bool = load_3d_descriptor_file(bool_path)
    print(f"  {len(vol_bool)} códigos")

    records, excluded = [], []
    print("Alineando multimodalidad EXPLÍCITAMENTE POR CODE...")
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
        if cid not in vol_float:
            reasons.append("missing_full_float_voxel")
        if cid not in vol_bool:
            reasons.append("missing_full_bool_voxel")

        desc4 = None if skey is None else smiles_descriptors(row[SMILES_COL])
        if desc4 is None and "invalid_smiles" not in reasons:
            reasons.append("invalid_smiles")
        affinities = pd.to_numeric(
            pd.Series([row.get("Affinity 7E3D"), row.get("Affinity 8DT2")]), errors="coerce"
        ).to_numpy(dtype=np.float32)
        if np.isnan(affinities).any():
            reasons.append("missing_affinity")

        img_path = resolve_image_path(dataset_dir, row.get("Image Path"), cid)
        if img_path is None or not img_path.exists():
            reasons.append("missing_image")

        if reasons:
            rr = row.to_dict()
            rr["ExclusionReason"] = ";".join(sorted(set(reasons)))
            excluded.append(rr)
            continue

        volume8 = np.concatenate([vol_bool[cid], vol_float[cid]], axis=-1).astype(np.float32)
        if volume8.shape != (25, 25, 25, 8):
            raise RuntimeError(f"Volumen incorrecto para {cid}: {volume8.shape}")
        records.append({
            "OriginalRow": int(row["OriginalRow"]),
            "Code": cid,
            "Common Name": row.get("Common Name"),
            "SMILES": row[SMILES_COL],
            "StructureKey": skey,
            "LD50": float(row[TARGET_COL]),
            "ImagePath": str(img_path),
            "Descriptors": np.concatenate([desc4, affinities]).astype(np.float32),
            "Volume8": volume8,
        })

    pd.DataFrame(excluded).to_csv(audit_dir / "excluded_rows_all_reasons.csv", index=False)
    if not records:
        raise RuntimeError("No quedaron registros válidos.")

    meta = pd.DataFrame([{k: v for k, v in r.items() if k not in ("Descriptors", "Volume8")} for r in records])
    meta.to_csv(audit_dir / "retained_rows_metadata.csv", index=False)

    print(f"Filas originales: {len(df)}")
    print(f"Filas retenidas: {len(records)}")
    print(f"Estructuras únicas retenidas: {meta['StructureKey'].nunique()}")
    print(f"Códigos ambiguos excluidos: {len(ambiguous_codes)}")
    print(f"Estructuras con target conflictivo excluidas: {len(conflict_keys)}")
    print(f"Volúmenes completos disponibles: float={len(vol_float)}, bool={len(vol_bool)}")

    images = []
    for i, r in enumerate(records, 1):
        if i == 1 or i % 25 == 0 or i == len(records):
            print(f"  imágenes: {i}/{len(records)}", end="\r", flush=True)
        images.append(load_image(dataset_dir, r["ImagePath"], r["Code"]))
    print()

    descriptors = np.stack([r["Descriptors"] for r in records]).astype(np.float32)
    volumes = np.stack([r["Volume8"] for r in records]).astype(np.float32)
    labels = np.asarray([r["LD50"] for r in records], dtype=np.float32)
    images = np.stack(images).astype(np.float32)

    print(f"Shapes finales: images={images.shape}, volume25={volumes.shape}, descriptors={descriptors.shape}, y={labels.shape}")
    return meta, descriptors, images, volumes, labels


class LD50Dataset(Dataset):
    def __init__(self, descriptors, images, volumes, labels):
        self.descriptors = torch.from_numpy(descriptors)
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        self.volumes = torch.from_numpy(volumes).permute(0, 4, 1, 2, 3).contiguous()
        self.labels = torch.from_numpy(labels).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.volumes[idx], self.descriptors[idx], self.labels[idx]


class MergeSingletonBatchSampler:
    """
    Batch sampler que conserva todas las muestras pero evita un último batch
    de tamaño 1. Es necesario porque la rama 3D contiene BatchNorm3d y un
    tensor [1, C, 1, 1, 1] no puede normalizarse en modo training.

    Ejemplo: 129 muestras con batch_size=64 -> 64 + 65, no 64 + 64 + 1.
    """
    def __init__(self, n_samples, batch_size, shuffle):
        self.n_samples = int(n_samples)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.n_samples).tolist()
        else:
            order = list(range(self.n_samples))

        batches = [
            order[i:i + self.batch_size]
            for i in range(0, self.n_samples, self.batch_size)
        ]

        if len(batches) >= 2 and len(batches[-1]) == 1:
            batches[-2].extend(batches[-1])
            batches.pop()

        yield from batches

    def __len__(self):
        if self.n_samples == 0:
            return 0
        n = (self.n_samples + self.batch_size - 1) // self.batch_size
        if self.n_samples > self.batch_size and self.n_samples % self.batch_size == 1:
            n -= 1
        return n


def make_loader(dataset, indices, batch_size, shuffle, device):
    subset = Subset(dataset, list(map(int, indices)))
    batch_sampler = MergeSingletonBatchSampler(len(subset), batch_size, shuffle)
    return DataLoader(
        subset,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def class_metrics(y_true, y_pred, threshold=2000.0):
    true_toxic = np.asarray(y_true) <= threshold
    pred_toxic = np.asarray(y_pred) <= threshold
    tp = int(np.sum(true_toxic & pred_toxic))
    fn = int(np.sum(true_toxic & ~pred_toxic))
    fp = int(np.sum(~true_toxic & pred_toxic))
    tn = int(np.sum(~true_toxic & ~pred_toxic))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) else float("nan")
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "precision": precision, "recall": recall, "F1": f1, "accuracy": accuracy}


def train_fold(dataset, tr_idx, va_idx, hp, fold_dir, device, fold_seed):
    lr = float(hp.get("learning_rate", 5e-6))
    epochs = int(hp.get("epochs", 1000))
    batch_size = int(hp.get("batch_size", 64))
    # Defaults deliberately preserve the original protocol.
    use_amp = bool(int(hp.get("use_amp", 0))) and device.type == "cuda"
    early_patience = int(hp.get("early_stopping_patience", 0))  # 0 = disabled

    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(fold_seed)

    train_loader = make_loader(dataset, tr_idx, batch_size, True, device)
    val_loader = make_loader(dataset, va_idx, batch_size, False, device)

    model = FlexibleDenseNet(growth_rate=32, num_classes=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=100)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    val_losses = []
    best_path = fold_dir / "best_model.pth"
    final_path = fold_dir / "final_model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        n_train_batches = 0
        for image, volume, descriptors, label in train_loader:
            image = image.to(device, non_blocking=True)
            volume = volume.to(device, non_blocking=True)
            descriptors = descriptors.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(image=image, volume_small=None, volume_large=volume, descriptors=descriptors)
                loss = criterion(out, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_sum += float(loss.item())
            n_train_batches += 1

        model.eval()
        val_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for image, volume, descriptors, label in val_loader:
                image = image.to(device, non_blocking=True)
                volume = volume.to(device, non_blocking=True)
                descriptors = descriptors.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(image=image, volume_small=None, volume_large=volume, descriptors=descriptors)
                    loss = criterion(out, label)
                val_sum += float(loss.item())
                n_val_batches += 1
        val_loss = val_sum / max(n_val_batches, 1)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1

        if epoch == 1 or epoch % 25 == 0:
            train_loss = train_sum / max(n_train_batches, 1)
            print(f"      epoch {epoch:4d}/{epochs} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f} | best {best_loss:.4f} @ {best_epoch}")

        if early_patience > 0 and no_improve >= early_patience:
            print(f"      early stopping en epoch {epoch}; mejor epoch={best_epoch}")
            break

    torch.save(model.state_dict(), final_path)
    pd.DataFrame({"epoch": np.arange(1, len(val_losses)+1), "validation_mse": val_losses}).to_csv(fold_dir / "validation_loss.csv", index=False)
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, len(val_losses)+1), val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title("Validation loss")
    plt.tight_layout()
    plt.savefig(fold_dir / "validation_loss.png", dpi=140)
    plt.close()

    return best_path, final_path, best_loss, best_epoch, len(val_losses)


def ensemble_predict(dataset, indices, model_paths, batch_size, device):
    loader = make_loader(dataset, indices, batch_size, False, device)
    models = []
    for path in model_paths:
        m = FlexibleDenseNet(growth_rate=32, num_classes=1).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
    preds, trues = [], []
    with torch.no_grad():
        for image, volume, descriptors, label in loader:
            image = image.to(device, non_blocking=True)
            volume = volume.to(device, non_blocking=True)
            descriptors = descriptors.to(device, non_blocking=True)
            fold_preds = [m(image=image, volume_small=None, volume_large=volume, descriptors=descriptors) for m in models]
            mean_pred = torch.stack(fold_preds, dim=0).mean(dim=0)
            preds.extend(mean_pred.detach().cpu().numpy().reshape(-1).tolist())
            trues.extend(label.numpy().reshape(-1).tolist())
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(trues, dtype=float), np.asarray(preds, dtype=float)


def summarize_predictions(meta_subset, true, pred, out_csv_prefix):
    pred_df = meta_subset.reset_index(drop=True).copy()
    pred_df["LD50_real"] = true
    pred_df["LD50_predicho"] = pred
    pred_df["AbsoluteError"] = np.abs(true - pred)
    pred_df.to_csv(str(out_csv_prefix) + "_row_level.csv", index=False)

    struct_df = (pred_df.groupby("StructureKey", as_index=False)
                 .agg(Common_Name=("Common Name", lambda s: ";".join(sorted(set(map(str, s.dropna()))))),
                      LD50_real=("LD50_real", "mean"),
                      LD50_predicho=("LD50_predicho", "mean"),
                      n_rows=("Code", "size")))
    struct_df["AbsoluteError"] = np.abs(struct_df["LD50_real"] - struct_df["LD50_predicho"])
    struct_df.to_csv(str(out_csv_prefix) + "_structure_level.csv", index=False)

    row_mse = mean_squared_error(true, pred)
    struct_mse = mean_squared_error(struct_df["LD50_real"], struct_df["LD50_predicho"])
    metrics = {
        "row_MAE": float(mean_absolute_error(true, pred)),
        "row_MSE": float(row_mse),
        "row_RMSE": float(np.sqrt(row_mse)),
        "structure_MAE": float(mean_absolute_error(struct_df["LD50_real"], struct_df["LD50_predicho"])),
        "structure_MSE": float(struct_mse),
        "structure_RMSE": float(np.sqrt(struct_mse)),
    }
    metrics.update({f"row_class_{k}": v for k, v in class_metrics(true, pred).items()})
    metrics.update({f"structure_class_{k}": v for k, v in class_metrics(struct_df["LD50_real"], struct_df["LD50_predicho"]).items()})
    return metrics


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    results_root = Path(args.results_dir)
    audit_dir = results_root / "audit"
    results_root.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    hp = load_hyperparams("hyperparams.txt")
    k_folds = int(hp.get("k_folds", 5))
    batch_size = int(hp.get("batch_size", 64))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    print(f"Protocolo: lr={hp.get('learning_rate', 5e-6)}, epochs={int(hp.get('epochs',1000))}, batch={batch_size}, folds={k_folds}")
    print(f"AMP={'ON' if bool(int(hp.get('use_amp',0))) else 'OFF'}; early stopping={int(hp.get('early_stopping_patience',0)) or 'OFF'}")

    meta, descriptors, images, volumes, labels = build_dataset(dataset_dir, audit_dir)
    dataset = LD50Dataset(descriptors, images, volumes, labels)

    tests = [args.test] if args.test is not None else list(range(1, N_TESTS + 1))
    split_rows = []
    split_summary = []
    for test_no in tests:
        if test_no < 1 or test_no > 10:
            raise ValueError("--test debe estar entre 1 y 10")
        test_file = dataset_dir / f"test_{test_no}_updated.csv"
        raw = pd.read_csv(test_file)
        original_codes = {norm_code(v) for v in raw[ID_COL]}
        original_codes.discard(None)
        seed_groups = set(meta.loc[meta[ID_COL].isin(original_codes), "StructureKey"])
        if not seed_groups:
            raise RuntimeError(f"TEST {test_no}: ningún Code del CSV sobrevivió al filtrado")
        test_mask = meta["StructureKey"].isin(seed_groups)
        te = np.flatnonzero(test_mask.to_numpy())
        te_groups = set(meta.iloc[te]["StructureKey"])
        tr = np.flatnonzero((~meta["StructureKey"].isin(te_groups)).to_numpy())
        tr_groups = set(meta.iloc[tr]["StructureKey"])
        overlap = tr_groups & te_groups
        if overlap:
            raise RuntimeError(f"TEST {test_no}: leakage train-test ({len(overlap)} estructuras)")
        split_summary.append({
            "test": test_no, "train_rows": len(tr), "test_rows": len(te),
            "train_unique_structures": len(tr_groups), "test_unique_structures": len(te_groups),
            "train_test_overlap": len(overlap),
        })

        gv = meta.iloc[tr]["StructureKey"].to_numpy()
        gkf = GroupKFold(n_splits=k_folds)
        for fold, (trl, val) in enumerate(gkf.split(tr, groups=gv), 1):
            trg = tr[trl]; vag = tr[val]
            s_tr = set(meta.iloc[trg]["StructureKey"])
            s_va = set(meta.iloc[vag]["StructureKey"])
            split_rows.append({
                "test": test_no, "fold": fold,
                "train_rows": len(trg), "validation_rows": len(vag), "test_rows": len(te),
                "train_structures": len(s_tr), "validation_structures": len(s_va), "test_structures": len(te_groups),
                "train_validation_overlap": len(s_tr & s_va),
                "train_test_overlap": len(s_tr & te_groups),
                "validation_test_overlap": len(s_va & te_groups),
            })

    pd.DataFrame(split_summary).to_csv(audit_dir / "split_summary.csv", index=False)
    pd.DataFrame(split_rows).to_csv(audit_dir / "split_audit_all_folds.csv", index=False)
    print("\n=== AUDITORÍA DE SPLITS ===")
    print(pd.DataFrame(split_summary).to_string(index=False))
    if max(max(r["train_validation_overlap"], r["train_test_overlap"], r["validation_test_overlap"]) for r in split_rows) != 0:
        raise RuntimeError("Se detectó leakage en auditoría")
    print("Todos los solapamientos estructurales train/validation/test = 0")

    if args.audit_only:
        print("\nAuditoría terminada. No se ha iniciado entrenamiento.")
        return

    all_best_metrics = []
    all_final_metrics = []
    all_best_preds = []

    for test_no in tests:
        exp_dir = results_root / f"test_{test_no:02d}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Reanudación: si este test ya terminó por completo, reutilizar lo guardado.
        best_metrics_path = exp_dir / "metrics_best_checkpoint.json"
        final_metrics_path = exp_dir / "metrics_final_epoch.json"
        best_pred_path = exp_dir / "predictions_best_checkpoint_row_level.csv"
        if best_metrics_path.exists() and final_metrics_path.exists():
            with open(best_metrics_path, "r", encoding="utf-8") as f:
                existing_best = json.load(f)
            with open(final_metrics_path, "r", encoding="utf-8") as f:
                existing_final = json.load(f)
            all_best_metrics.append(existing_best)
            all_final_metrics.append(existing_final)
            if best_pred_path.exists():
                bp = pd.read_csv(best_pred_path)
                if "test" not in bp.columns:
                    bp.insert(0, "test", test_no)
                all_best_preds.append(bp)
            print("\n" + "="*80)
            print(f"TEST {test_no}: ya completado -> se reutilizan resultados guardados y se continúa.")
            print("="*80)
            continue

        raw = pd.read_csv(dataset_dir / f"test_{test_no}_updated.csv")
        original_codes = {norm_code(v) for v in raw[ID_COL]}; original_codes.discard(None)
        test_groups = set(meta.loc[meta[ID_COL].isin(original_codes), "StructureKey"])
        te = np.flatnonzero(meta["StructureKey"].isin(test_groups).to_numpy())
        test_groups = set(meta.iloc[te]["StructureKey"])
        tr = np.flatnonzero((~meta["StructureKey"].isin(test_groups)).to_numpy())
        train_groups = set(meta.iloc[tr]["StructureKey"])

        effective = meta.iloc[te].copy()
        effective["WasListedByOriginalCode"] = effective[ID_COL].isin(original_codes)
        effective.to_csv(exp_dir / "effective_test_rows.csv", index=False)

        print("\n" + "="*80)
        print(f"TEST {test_no}: train rows={len(tr)}, test rows={len(te)}, train structures={len(train_groups)}, test structures={len(test_groups)}, overlap=0")
        print("="*80)

        gv = meta.iloc[tr]["StructureKey"].to_numpy()
        gkf = GroupKFold(n_splits=k_folds)
        best_paths, final_paths, fold_info = [], [], []
        for fold, (trl, val) in enumerate(gkf.split(tr, groups=gv), 1):
            trg = tr[trl]; vag = tr[val]
            s_tr = set(meta.iloc[trg]["StructureKey"]); s_va = set(meta.iloc[vag]["StructureKey"])
            if s_tr & s_va or s_tr & test_groups or s_va & test_groups:
                raise RuntimeError(f"Leakage detectado en TEST {test_no}, fold {fold}")
            fold_dir = exp_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            rem = len(trg) % batch_size
            if rem == 1 and len(trg) > batch_size:
                print(f"  Fold {fold}/{k_folds}: train rows={len(trg)}, val rows={len(vag)}, group overlap=0 | batch final singleton -> fusionado con batch anterior")
            else:
                print(f"  Fold {fold}/{k_folds}: train rows={len(trg)}, val rows={len(vag)}, group overlap=0")
            best_p, final_p, best_loss, best_epoch, epochs_done = train_fold(
                dataset, trg, vag, hp, fold_dir, device, RANDOM_STATE + test_no*100 + fold
            )
            best_paths.append(best_p); final_paths.append(final_p)
            fold_info.append({"fold": fold, "best_val_mse": best_loss, "best_epoch": best_epoch, "epochs_completed": epochs_done})
        pd.DataFrame(fold_info).to_csv(exp_dir / "fold_training_summary.csv", index=False)

        y_true, pred_best = ensemble_predict(dataset, te, best_paths, batch_size, device)
        best_metrics = summarize_predictions(effective, y_true, pred_best, exp_dir / "predictions_best_checkpoint")
        best_metrics.update({
            "test": test_no, "train_rows": len(tr), "test_rows": len(te),
            "train_unique_structures": len(train_groups), "test_unique_structures": len(test_groups),
            "train_test_structure_overlap": 0,
        })
        with open(exp_dir / "metrics_best_checkpoint.json", "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2, allow_nan=True)
        all_best_metrics.append(best_metrics)
        bp = pd.read_csv(exp_dir / "predictions_best_checkpoint_row_level.csv")
        bp.insert(0, "test", test_no); all_best_preds.append(bp)

        y_true2, pred_final = ensemble_predict(dataset, te, final_paths, batch_size, device)
        final_metrics = summarize_predictions(effective, y_true2, pred_final, exp_dir / "predictions_final_epoch")
        final_metrics.update({
            "test": test_no, "train_rows": len(tr), "test_rows": len(te),
            "train_unique_structures": len(train_groups), "test_unique_structures": len(test_groups),
            "train_test_structure_overlap": 0,
        })
        with open(exp_dir / "metrics_final_epoch.json", "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2, allow_nan=True)
        all_final_metrics.append(final_metrics)

        print(f"  -> BEST checkpoint: MAE row={best_metrics['row_MAE']:.3f}; MSE row={best_metrics['row_MSE']:.3f}; MAE estructura={best_metrics['structure_MAE']:.3f}")
        print(f"  -> FINAL epoch:     MAE row={final_metrics['row_MAE']:.3f}; MSE row={final_metrics['row_MSE']:.3f}; MAE estructura={final_metrics['structure_MAE']:.3f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_df = pd.DataFrame(all_best_metrics).sort_values("test")
    final_df = pd.DataFrame(all_final_metrics).sort_values("test")
    best_df.to_csv(results_root / "metrics_by_test_BEST_CHECKPOINT.csv", index=False)
    final_df.to_csv(results_root / "metrics_by_test_FINAL_EPOCH.csv", index=False)
    if all_best_preds:
        pd.concat(all_best_preds, ignore_index=True).to_csv(results_root / "predictions_all_tests_BEST_CHECKPOINT.csv", index=False)

    def summary(df):
        return {
            "n_tests": int(len(df)),
            "row_MAE_mean": float(df["row_MAE"].mean()),
            "row_MAE_std": float(df["row_MAE"].std(ddof=1)),
            "row_MSE_mean": float(df["row_MSE"].mean()),
            "row_MSE_std": float(df["row_MSE"].std(ddof=1)),
            "structure_MAE_mean": float(df["structure_MAE"].mean()),
            "structure_MAE_std": float(df["structure_MAE"].std(ddof=1)),
            "structure_MSE_mean": float(df["structure_MSE"].mean()),
            "structure_MSE_std": float(df["structure_MSE"].std(ddof=1)),
            "row_precision_mean": float(df["row_class_precision"].mean()),
            "row_recall_mean": float(df["row_class_recall"].mean()),
            "row_F1_mean": float(df["row_class_F1"].mean()),
            "row_accuracy_mean": float(df["row_class_accuracy"].mean()),
            "max_train_test_structure_overlap": int(df["train_test_structure_overlap"].max()),
        }

    best_summary = summary(best_df); final_summary = summary(final_df)
    with open(results_root / "metrics_summary_BEST_CHECKPOINT.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2, allow_nan=True)
    with open(results_root / "metrics_summary_FINAL_EPOCH.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, allow_nan=True)

    print("\n=== RESUMEN FINAL - BEST VALIDATION CHECKPOINTS (PRIMARIO) ===")
    print(json.dumps(best_summary, indent=2, allow_nan=True))
    print("\n=== RESUMEN FINAL - FINAL EPOCH (COMPARACIÓN CON CÓDIGO ANTIGUO) ===")
    print(json.dumps(final_summary, indent=2, allow_nan=True))
    print(f"\nResultados: {results_root.resolve()}")


if __name__ == "__main__":
    main()
