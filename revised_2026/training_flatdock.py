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
