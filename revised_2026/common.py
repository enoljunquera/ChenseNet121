from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, inchi
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset

ID_COL = "Code"
SMILES_COL = "SMILES"
TARGET_COL = "Acute Oral LD50"
RANDOM_STATE = 42
THRESHOLD = 2000.0


def norm_code(v):
    if pd.isna(v): return None
    s = str(v).strip()
    if not s or s.lower() == "code": return None
    if re.fullmatch(r"\d+\.0", s): s = s[:-2]
    return s


def structure_key(smiles):
    if pd.isna(smiles): return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return None
    try:
        key = inchi.MolToInchiKey(mol)
        if key: return key
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def smiles_descriptors(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return None
    return np.asarray([
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumRotatableBonds(mol),
    ], dtype=np.float32)


def load_descriptor_file(path: Path, spatial_size: int):
    expected = spatial_size ** 3 * 4
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    out = {}
    for block in content.strip().split("**************************"):
        lines = block.strip().splitlines()
        if len(lines) < 2: continue
        cid = norm_code(lines[0])
        if cid is None: continue
        numeric = " ".join(lines[1:]).replace("[", " ").replace("]", " ").replace("\t", " ")
        toks = [t for t in re.sub(r"\s+", " ", numeric).split(" ") if t]
        vals = np.asarray(list(map(float, toks)), dtype=np.float32)
        if vals.size != expected:
            raise ValueError(f"{path.name}, Code={cid}: expected {expected} values, got {vals.size}")
        out[cid] = vals.reshape(spatial_size, spatial_size, spatial_size, 4)
    return out


def available_codes_from_descriptor_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    out = set()
    for block in content.strip().split("**************************"):
        lines = block.strip().splitlines()
        if not lines: continue
        cid = norm_code(lines[0])
        if cid: out.add(cid)
    return out


def resolve_image_path(dataset_dir: Path, raw_path, code):
    candidates = []
    if not pd.isna(raw_path):
        s = str(raw_path).strip().replace("\\", "/")
        if s.startswith("./"): s = s[2:]
        candidates += [dataset_dir / s, Path(str(raw_path))]
    if code: candidates.append(dataset_dir / "images" / f"CID_{code}.png")
    for p in candidates:
        if p.exists(): return p
    return None


def load_image(path: Path, target_size=(300, 300)):
    try: resample = Image.Resampling.NEAREST
    except AttributeError: resample = Image.NEAREST
    with Image.open(path) as im:
        im = im.convert("RGB").resize(target_size, resample=resample)
        return np.asarray(im, dtype=np.float32) / 255.0


def build_common_cohort(dataset_dir: Path, load_arrays=True, require_small=True):
    """Rebuild EXACTLY the leakage-safe 249-row / 199-structure cohort used in the 3-fold benchmark."""
    combined_path = dataset_dir / "combined_dataset.csv"
    large_bool_path = dataset_dir / "dataset_descriptores_bool.txt"
    large_float_path = dataset_dir / "dataset_descriptores_float.txt"
    small_bool_path = dataset_dir / "dataset_descriptores_2x2x2_bool.txt"
    small_float_path = dataset_dir / "dataset_descriptores_2x2x2_float.txt"
    for p in [combined_path, large_bool_path, large_float_path, small_bool_path, small_float_path]:
        if not p.exists(): raise FileNotFoundError(p)

    df = pd.read_csv(combined_path).copy()
    df["OriginalRow"] = np.arange(len(df))
    df[ID_COL] = df[ID_COL].map(norm_code)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df["StructureKey"] = df[SMILES_COL].map(structure_key)

    code_n_struct = df.dropna(subset=[ID_COL, "StructureKey"]).groupby(ID_COL)["StructureKey"].nunique()
    ambiguous_codes = set(code_n_struct[code_n_struct > 1].index)
    conflict_keys = set()
    for key, grp in df.dropna(subset=["StructureKey"]).groupby("StructureKey"):
        if grp[TARGET_COL].dropna().nunique() > 1:
            conflict_keys.add(key)

    lb_codes = available_codes_from_descriptor_file(large_bool_path)
    lf_codes = available_codes_from_descriptor_file(large_float_path)
    sb_codes = available_codes_from_descriptor_file(small_bool_path)
    sf_codes = available_codes_from_descriptor_file(small_float_path)

    records = []
    for _, row in df.iterrows():
        cid = row[ID_COL]; skey = row["StructureKey"]
        if cid is None or skey is None or pd.isna(row[TARGET_COL]): continue
        if cid in ambiguous_codes or skey in conflict_keys: continue
        if cid not in lb_codes or cid not in lf_codes: continue
        if require_small and (cid not in sb_codes or cid not in sf_codes): continue
        desc4 = smiles_descriptors(row[SMILES_COL])
        if desc4 is None: continue
        aff = pd.to_numeric(pd.Series([row.get("Affinity 7E3D"), row.get("Affinity 8DT2")]), errors="coerce").to_numpy(dtype=np.float32)
        if np.isnan(aff).any(): continue
        img = resolve_image_path(dataset_dir, row.get("Image Path"), cid)
        if img is None: continue
        records.append({
            "OriginalRow": int(row["OriginalRow"]), "Code": cid, "Common Name": row.get("Common Name"),
            "SMILES": row[SMILES_COL], "StructureKey": skey, "LD50": float(row[TARGET_COL]),
            "ImagePath": str(img), "Desc4": desc4, "Aff2": aff,
        })

    meta = pd.DataFrame([{k:v for k,v in r.items() if k not in {"Desc4","Aff2"}} for r in records])
    if len(meta) != 249 or meta["StructureKey"].nunique() != 199:
        raise RuntimeError(f"Cohort mismatch: got {len(meta)} rows / {meta['StructureKey'].nunique()} structures; expected 249 / 199")

    desc4 = np.stack([r["Desc4"] for r in records]).astype(np.float32)
    aff2 = np.stack([r["Aff2"] for r in records]).astype(np.float32)
    y = np.asarray([r["LD50"] for r in records], dtype=np.float32)

    images = volumes = None
    if load_arrays:
        lb = load_descriptor_file(large_bool_path, 25)
        lf = load_descriptor_file(large_float_path, 25)
        images = np.stack([load_image(Path(r["ImagePath"])) for r in records]).astype(np.float32)
        volumes = np.stack([np.concatenate([lb[r["Code"]], lf[r["Code"]]], axis=-1) for r in records]).astype(np.float32)
    return meta, desc4, aff2, images, volumes, y


def tox_bin(y):
    return np.digitize(np.asarray(y, dtype=float), bins=[50.,500.,2000.], right=True)


def make_threefold_splits(meta: pd.DataFrame):
    struct = meta.groupby("StructureKey", as_index=False).agg(LD50=("LD50","mean"), n_rows=("Code","size"))
    struct["tox_bin"] = tox_bin(struct["LD50"])
    counts = struct["tox_bin"].value_counts()
    if len(counts) >= 2 and counts.min() >= 3:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(struct, struct["tox_bin"])
    else:
        splitter = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(struct)
    outer=[]
    for fold_no,(train_si,test_si) in enumerate(split_iter,1):
        test_keys=set(struct.iloc[test_si]["StructureKey"])
        outer_train=struct.iloc[train_si].reset_index(drop=True)
        bins=outer_train["tox_bin"].to_numpy()
        try:
            sss=StratifiedShuffleSplit(n_splits=1,test_size=.20,random_state=RANDOM_STATE+fold_no)
            tr_si,va_si=next(sss.split(outer_train,bins))
        except ValueError:
            rng=np.random.default_rng(RANDOM_STATE+fold_no); perm=rng.permutation(len(outer_train)); nval=max(1,int(round(.20*len(outer_train))))
            va_si=perm[:nval]; tr_si=perm[nval:]
        train_keys=set(outer_train.iloc[tr_si]["StructureKey"]); val_keys=set(outer_train.iloc[va_si]["StructureKey"])
        if train_keys&val_keys or train_keys&test_keys or val_keys&test_keys: raise RuntimeError("split overlap")
        outer.append({
            "fold":fold_no,
            "train_idx":np.flatnonzero(meta["StructureKey"].isin(train_keys).to_numpy()),
            "val_idx":np.flatnonzero(meta["StructureKey"].isin(val_keys).to_numpy()),
            "test_idx":np.flatnonzero(meta["StructureKey"].isin(test_keys).to_numpy()),
            "train_keys":train_keys,"val_keys":val_keys,"test_keys":test_keys,
        })
    coverage=pd.Series([k for o in outer for k in o["test_keys"]]).value_counts()
    if len(coverage)!=199 or not (coverage==1).all(): raise RuntimeError("3-fold coverage mismatch")
    return outer


class SignalDataset(Dataset):
    def __init__(self, desc4, aff2, images, volumes, y):
        self.desc4=torch.from_numpy(desc4)
        self.aff2=torch.from_numpy(aff2)
        self.images=torch.from_numpy(images).permute(0,3,1,2).contiguous() if images is not None else None
        self.volumes=torch.from_numpy(volumes).permute(0,4,1,2,3).contiguous() if volumes is not None else None
        self.y=torch.from_numpy(y).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        image = self.images[i] if self.images is not None else torch.empty(0)
        volume = self.volumes[i] if self.volumes is not None else torch.empty(0)
        return image, volume, self.desc4[i], self.aff2[i], self.y[i]


class MergeSingletonBatchSampler:
    def __init__(self,n_samples,batch_size,shuffle): self.n_samples=int(n_samples); self.batch_size=int(batch_size); self.shuffle=bool(shuffle)
    def __iter__(self):
        order=torch.randperm(self.n_samples).tolist() if self.shuffle else list(range(self.n_samples))
        batches=[order[i:i+self.batch_size] for i in range(0,self.n_samples,self.batch_size)]
        if len(batches)>=2 and len(batches[-1])==1:
            batches[-2].extend(batches[-1]); batches.pop()
        yield from batches
    def __len__(self):
        if self.n_samples==0: return 0
        n=math.ceil(self.n_samples/self.batch_size)
        if self.n_samples>self.batch_size and self.n_samples%self.batch_size==1: n-=1
        return n


def make_loader(dataset, indices, batch_size, shuffle, device):
    subset=Subset(dataset,list(map(int,indices)))
    sampler=MergeSingletonBatchSampler(len(subset),batch_size,shuffle)
    return DataLoader(subset,batch_sampler=sampler,num_workers=0,pin_memory=(device.type=="cuda"))


def class_metrics(y_true,y_pred):
    t=np.asarray(y_true)<=THRESHOLD; p=np.asarray(y_pred)<=THRESHOLD
    tp=int(np.sum(t&p)); fn=int(np.sum(t&~p)); fp=int(np.sum(~t&p)); tn=int(np.sum(~t&~p))
    precision=tp/(tp+fp) if tp+fp else float("nan"); recall=tp/(tp+fn) if tp+fn else float("nan")
    f1=2*precision*recall/(precision+recall) if np.isfinite(precision) and np.isfinite(recall) and precision+recall else float("nan")
    return {"TP":tp,"FN":fn,"FP":fp,"TN":tn,"precision":precision,"recall":recall,"F1":f1,"accuracy":(tp+tn)/max(tp+fn+fp+tn,1)}


def prediction_metrics(meta_test,true,pred):
    row=meta_test.reset_index(drop=True).copy(); row["LD50_real"]=np.asarray(true,float); row["LD50_predicho"]=np.asarray(pred,float)
    row["AbsoluteError"]=np.abs(row["LD50_real"]-row["LD50_predicho"])
    st=(row.groupby("StructureKey",as_index=False).agg(Common_Name=("Common Name",lambda s:";".join(sorted(set(map(str,s.dropna()))))),LD50_real=("LD50_real","mean"),LD50_predicho=("LD50_predicho","mean"),n_rows=("Code","size")))
    st["AbsoluteError"]=np.abs(st["LD50_real"]-st["LD50_predicho"])
    rmse=mean_squared_error(row["LD50_real"],row["LD50_predicho"]); smse=mean_squared_error(st["LD50_real"],st["LD50_predicho"])
    m={"row_MAE":float(mean_absolute_error(row["LD50_real"],row["LD50_predicho"])),"row_MSE":float(rmse),"row_RMSE":float(np.sqrt(rmse)),
       "structure_MAE":float(mean_absolute_error(st["LD50_real"],st["LD50_predicho"])),"structure_MSE":float(smse),"structure_RMSE":float(np.sqrt(smse))}
    m.update({f"row_class_{k}":v for k,v in class_metrics(row["LD50_real"],row["LD50_predicho"]).items()})
    m.update({f"structure_class_{k}":v for k,v in class_metrics(st["LD50_real"],st["LD50_predicho"]).items()})
    return row,st,m
