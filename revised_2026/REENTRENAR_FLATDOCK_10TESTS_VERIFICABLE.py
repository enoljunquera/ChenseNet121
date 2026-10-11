"""FlatDock QSAR benchmark, reproducible and structure-grouped.

Run from the LD50 project root (alongside 2026_3_correccion_outlier and 2026_4).
Does not change original experiment outputs. Does not retrain multimodal CNNs.

Targets the *three classical QSAR models* (Ridge, LinearSVR, RandomForest) using
D + A + F (4 molecular descriptors, 2 affinities, 17 FlatDock descriptors).
Use --with-controls for the comparison with D and D+A under the identical protocol.

Uses exact held-out STRUCTURES of the confirmed 2026_4 ChInceptionV3 tests, not
original test CSVs or any row-randomized split. Rebuilds the correct 249/199
cohort from the verified predictions and cross-checks against corrected data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

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

RANDOM_STATE = 42
TARGET = "Acute Oral LD50"
AFF_COLS = ["Affinity 7E3D", "Affinity 8DT2"]
N_ROWS = 249
N_STRUCTURES = 199


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--audit-only', action='store_true', help='Verify the 10-test split and cohort; fit no models')
    p.add_argument('--with-controls', action='store_true', help='Train D and D+A controls too')
    p.add_argument('--force', action='store_true', help='Overwrite existing results')
    return p.parse_args()


def code(v):
    s = str(v).strip()
    return s[:-2] if s.endswith('.0') and s[:-2].isdigit() else s


def inchikey(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    return inchi.MolToInchiKey(mol)


def descriptor4(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    return [Descriptors.MolWt(mol), Descriptors.TPSA(mol),
            Descriptors.MolLogP(mol), Descriptors.NumRotatableBonds(mol)]


def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def load_cohort(root):
    corrected=root/'2026_3_correccion_outlier'/'LD50_docking_signal_followup'/'dataset'
    combined_file=corrected/'combined_dataset.csv'
    flat_file=corrected/'voxel_features_nutridock_style_synthesized.csv'
    refroot=root/'2026_4'/'ChInceptionV3'
    for p in (combined_file, flat_file, refroot):
        if not p.exists(): raise FileNotFoundError(f'Not found: {p}')
    combined=pd.read_csv(combined_file)
    combined['Code']=combined['Code'].map(code)
    references={}
    all_tests=[]
    for i in range(1,11):
        p=refroot/f'test_{i:02d}'/'predictions_row_level.csv'
        if not p.exists(): raise FileNotFoundError(f'Missing canonical test: {p}')
        t=pd.read_csv(p)
        required={'OriginalRow','Code','SMILES','StructureKey','LD50_real'}
        if not required.issubset(t.columns):
            raise ValueError(f'Unexpected columns in {p}: {sorted(required-set(t.columns))}')
        t['Code']=t['Code'].map(code)
        if t.duplicated('OriginalRow').any(): raise ValueError(f'Repeated OriginalRow within test {i}')
        references[i]={'path':str(p),'hash':sha256(p)}
        t['test']=i
        all_tests.append(t)
    merged=pd.concat(all_tests,ignore_index=True)
    original_cols=['OriginalRow','Code','SMILES','StructureKey','LD50_real']
    for field in original_cols[1:]:
        n=merged.groupby('OriginalRow')[field].nunique(dropna=False)
        if (n>1).any(): raise ValueError(f'2026_4 gives inconsistent {field} for same OriginalRow')
    df=merged.drop_duplicates('OriginalRow').sort_values('OriginalRow').reset_index(drop=True)
    if len(df)!=N_ROWS or df.StructureKey.nunique()!=N_STRUCTURES:
        raise ValueError(f'Unexpected cohort: {len(df)} rows / {df.StructureKey.nunique()} structures')
    if df['OriginalRow'].min()<0 or df['OriginalRow'].max()>=len(combined):
        raise ValueError('OriginalRow index out of bounds')
    original=combined.iloc[df.OriginalRow.to_numpy()].reset_index(drop=True)
    if not (original.Code.to_numpy()==df.Code.to_numpy()).all():
        raise ValueError('Code disagreement 2026_4 vs corrected source')
    if not (original.SMILES.astype(str).to_numpy()==df.SMILES.astype(str).to_numpy()).all():
        raise ValueError('SMILES disagreement 2026_4 vs corrected source')
    yy=pd.to_numeric(original[TARGET],errors='coerce').to_numpy(float)
    if not np.allclose(yy,df.LD50_real.to_numpy(float),atol=1e-5,rtol=0):
        raise ValueError('LD50 disagreement 2026_4 vs corrected source; no training allowed')
    if any(inchikey(s)!=k for s,k in zip(df.SMILES,df.StructureKey)):
        raise ValueError('StructureKey mismatch against RDKit canonical InChIKey')
    if df.groupby('StructureKey').LD50_real.nunique().max()!=1:
        raise ValueError('Conflicting LD50 values within a structure')

    flat=pd.read_csv(flat_file)
    flat['Code']=flat['Code'].map(code)
    flatcols=[c for c in flat.columns if c!='Code']
    if len(flatcols)!=17:
        raise ValueError(f'Expected 17 FlatDock features, got {len(flatcols)}')
    if flat.Code.duplicated().any(): raise ValueError('Repeated FlatDock Code mapping')
    flat=flat.set_index('Code')
    missing=[v for v in df.Code if v not in flat.index]
    if missing: raise ValueError(f'Missing FlatDock codes: {missing[:10]}')
    Xflat=flat.loc[df.Code,flatcols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    Xd=np.asarray([descriptor4(s) for s in df.SMILES],float)
    Xa=original[AFF_COLS].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    if not (np.isfinite(Xflat).all() and np.isfinite(Xd).all() and np.isfinite(Xa).all()):
        raise ValueError('Non-finite features in curated cohort')
    y=df.LD50_real.to_numpy(float)
    xs={'DAF':np.column_stack([Xd,Xa,Xflat]),'D':Xd,'DA':np.column_stack([Xd,Xa])}
    test_indices={}
    audit=[]
    all_keys=set(df.StructureKey)
    for i,t in enumerate(all_tests,1):
        test_keys=set(t.StructureKey)
        te=np.flatnonzero(df.StructureKey.isin(test_keys).to_numpy())
        tr=np.flatnonzero(~df.StructureKey.isin(test_keys).to_numpy())
        assert set(df.iloc[tr].StructureKey).isdisjoint(set(df.iloc[te].StructureKey))
        if set(df.iloc[te].OriginalRow)!=set(t.OriginalRow):
            raise ValueError(f'Test {i}: row membership differs from canonical 2026_4 test')
        if len(te)!=len(t):
            raise ValueError(f'Test {i}: incomplete grouped membership')
        test_indices[i]=(tr,te)
        audit.append({'test':i,'train_rows':len(tr),'test_rows':len(te),
                      'train_structures':len(all_keys-test_keys),'test_structures':len(test_keys),
                      'exact_structure_overlap':0,'matched_2026_4':True})
    return df,y,xs,audit,test_indices,{'data':str(combined_file),'data_sha256':sha256(combined_file),
                'flatdock':str(flat_file),'flatdock_sha256':sha256(flat_file),
                'canonical_test_files':references,'feature_names':{'D':['MolWt','TPSA','MolLogP','NumRotatableBonds'],
                  'A':AFF_COLS,'F':flatcols}}


def wrapped(reg):
    return TransformedTargetRegressor(regressor=reg,func=np.log1p,inverse_func=np.expm1,check_inverse=False)


def models():
    return {
      'Ridge':Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',wrapped(Ridge(alpha=1.0)))]),
      'LinearSVR':Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',wrapped(LinearSVR(C=1.0,epsilon=0.0,max_iter=50000,random_state=RANDOM_STATE)))]),
      'RandomForest':Pipeline([('imputer',SimpleImputer(strategy='median')),('model',wrapped(RandomForestRegressor(n_estimators=500,min_samples_leaf=1,n_jobs=-1,random_state=RANDOM_STATE)))])
    }


def compute_metrics(true,pred):
    true=np.asarray(true,float);pred=np.asarray(pred,float)
    pos=true<=2000;pp=pred<=2000
    tp=int((pos & pp).sum());fn=int((pos & ~pp).sum());fp=int((~pos & pp).sum());tn=int((~pos & ~pp).sum())
    precision=tp/(tp+fp) if tp+fp else float('nan')
    recall=tp/(tp+fn) if tp+fn else float('nan')
    f1=2*precision*recall/(precision+recall) if (precision+recall)>0 else float('nan')
    return {'MSE':float(mean_squared_error(true,pred)),'MAE':float(mean_absolute_error(true,pred)),
            'RMSE':float(np.sqrt(mean_squared_error(true,pred))),
            'TP':tp,'FN':fn,'FP':fp,'TN':tn,'precision':precision,'recall':recall,'F1':f1,
            'accuracy':float((tp+tn)/len(true))}


def main():
    a=args();root=a.root.resolve()
    print('LD50 root:',root)
    cohort,y,xs,audit,idx,source=load_cohort(root)
    out=root/'2026_6_flatdock_10tests_grouped'
    print(f'Curated cohort: {len(cohort)} rows, {cohort.StructureKey.nunique()} unique structures')
    for q in audit: print(f"  test {q['test']:2d}: train={q['train_rows']} rows/{q['train_structures']} structures; "
                          f"test={q['test_rows']} rows/{q['test_structures']} structures, leakage=0")
    out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(audit).to_csv(out/'SPLIT_AUDIT.csv',index=False)
    (out/'INPUT_MANIFEST.json').write_text(json.dumps(source,indent=2,ensure_ascii=False),encoding='utf-8')
    if a.audit_only:
        print('AUDIT PASSED: no models have been trained.');return

    settings=['DAF']+(['D','DA'] if a.with_controls else [])
    per_test=[]
    for setting in settings:
      X=xs[setting]
      for modelname,base in models().items():
        modeldir=out/f'{modelname}__{setting}'
        for testnum in range(1,11):
            tr,te=idx[testnum]
            dst=modeldir/f'test_{testnum:02d}'
            paths=[dst/'predictions_row_level.csv',dst/'predictions_structure_level.csv',dst/'metrics.json']
            if any(p.exists() for p in paths) and not a.force:
                if all(p.exists() for p in paths):
                    print('SKIP already complete:',modelname,setting,testnum)
                    per_test.append(json.loads(paths[-1].read_text(encoding='utf-8')))
                    continue
                raise RuntimeError(f'Partial results exist in {dst}; inspect or use --force')
            dst.mkdir(exist_ok=True,parents=True)
            est=clone(base)
            est.fit(X[tr],y[tr])
            pred=np.clip(np.asarray(est.predict(X[te]),float).reshape(-1),0,None)
            row=cohort.iloc[te].copy()
            row['LD50_predicho']=pred
            row['AbsoluteError']=np.abs(y[te]-pred)
            row['SquaredError']=(y[te]-pred)**2
            row=row.rename(columns={'LD50_real':'LD50_real'})
            st=row.groupby('StructureKey',as_index=False).agg(
                LD50_real=('LD50_real','mean'),LD50_predicho=('LD50_predicho','mean'),n_rows=('Code','size'))
            st['AbsoluteError']=np.abs(st.LD50_real-st.LD50_predicho)
            st['SquaredError']=(st.LD50_real-st.LD50_predicho)**2
            m={'setting':setting,'model':modelname,'test':testnum,'n_features':int(X.shape[1]),
              'train_rows':len(tr),'test_rows':len(te),
              'train_structures':cohort.iloc[tr].StructureKey.nunique(),
              'test_structures':st.shape[0],'train_test_structure_overlap':0}
            m.update({f'row_{k}':v for k,v in compute_metrics(y[te],pred).items()})
            m.update({f'structure_{k}':v for k,v in compute_metrics(st.LD50_real,st.LD50_predicho).items()})
            row.to_csv(paths[0],index=False)
            st.to_csv(paths[1],index=False)
            paths[2].write_text(json.dumps(m,indent=2,ensure_ascii=False,allow_nan=True),encoding='utf-8')
            per_test.append(m)
            print(f"{modelname:12} {setting:3} test {testnum:2d}: MSE={m['structure_MSE']/1e6:.3f}e6; MAE={m['structure_MAE']:.1f}",flush=True)
    met=pd.DataFrame(per_test).sort_values(['setting','model','test'])
    met.to_csv(out/'METRICS_BY_TEST.csv',index=False)
    sums=[]
    for (setting,model),grp in met.groupby(['setting','model']):
        if len(grp)!=10: raise ValueError('Incomplete set in summary')
        d={'setting':setting,'model':model,'n_tests':10}
        for prefix in ('structure','row'):
            for metric in ('MSE','MAE'):
                values=grp[f'{prefix}_{metric}'].to_numpy(float)
                d[f'{prefix}_{metric}_mean']=values.mean()
                d[f'{prefix}_{metric}_median']=np.median(values)
                d[f'{prefix}_{metric}_Q1']=np.quantile(values,.25)
                d[f'{prefix}_{metric}_Q3']=np.quantile(values,.75)
                d[f'{prefix}_{metric}_IQR']=np.quantile(values,.75)-np.quantile(values,.25)
        sums.append(d)
    pd.DataFrame(sums).sort_values('structure_MSE_mean').to_csv(out/'METRICS_SUMMARY_10TEST.csv',index=False)
    # A strict reproducibility check is possible when --with-controls is used:
    # the D and DA pipelines must reproduce the existing 2026_4 baseline predictions.
    existing_names={'LinearSVR':{'D':'LinearSVR_NoDock','DA':'LinearSVR_affinity'},
                    'RandomForest':{'D':'RandomForest_NoDock','DA':'RandomForest_affinity'},
                    'Ridge':{'D':'Ridge_NoDock','DA':'Ridge_affinity'}}
    checks=[]
    if a.with_controls:
        for modelname,aliases in existing_names.items():
            for setting,oldname in aliases.items():
                for testnum in range(1,11):
                    oldfile=root/'2026_4'/oldname/f'test_{testnum:02d}'/'predictions_structure_level.csv'
                    newfile=out/f'{modelname}__{setting}'/f'test_{testnum:02d}'/'predictions_structure_level.csv'
                    if not oldfile.exists():
                        raise FileNotFoundError(f'Cannot verify D/DA reproduction: missing {oldfile}')
                    oldpred=pd.read_csv(oldfile)[['StructureKey','LD50_predicho']]
                    newpred=pd.read_csv(newfile)[['StructureKey','LD50_predicho']]
                    joined=newpred.merge(oldpred,on='StructureKey',how='outer',suffixes=('_new','_2026_4'),indicator=True)
                    if (joined._merge!='both').any():
                        raise ValueError(f'{modelname}/{setting}/test {testnum}: structure membership mismatch')
                    diffs=np.abs(joined.LD50_predicho_new-joined.LD50_predicho_2026_4)
                    mx=float(diffs.max())
                    checks.append({'model':modelname,'setting':setting,'test':testnum,
                        'max_abs_prediction_difference_mgkg':mx})
                    if mx>0.01:
                        raise ValueError(f'Baseline replication FAILED for {modelname} {setting} test {testnum}: diff={mx}')
        if len(checks) != 60:
            raise ValueError(f'Expected 60 control comparisons, got {len(checks)}')
        if checks:
            pd.DataFrame(checks).to_csv(out/'REPRODUCIBILITY_VS_2026_4.csv',index=False)
            print('PASS: Reproduced',len(checks),'D/DA control partitions from 2026_4 (<=0.01 mg/kg)')
    print('\nDONE! Results:',out)
    print('Primary table: METRICS_SUMMARY_10TEST.csv')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print('\nSTOPPED WITHOUT SILENT RECOVERY:',exc,file=sys.stderr)
        raise
