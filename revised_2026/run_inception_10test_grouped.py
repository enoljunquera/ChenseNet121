from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold

from common import build_common_cohort,SignalDataset,make_loader,prediction_metrics,norm_code,RANDOM_STATE
from models.inceptionV3 import InceptionV3_Flexible

def parse_args():
    p=argparse.ArgumentParser();p.add_argument('--dataset-dir',default='./dataset');p.add_argument('--results-dir',default='./results/inception_10test_grouped');p.add_argument('--epochs',type=int,default=1000);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--learning-rate',type=float,default=5e-6);p.add_argument('--test',type=int,choices=range(1,11),default=None);p.add_argument('--audit-only',action='store_true');return p.parse_args()

def batch_inputs(batch,device):
    im,vo,d4,aff,y=batch;return im.to(device),vo.to(device),torch.cat([d4,aff],1).to(device),y.to(device)

def train_fold(ds,tr,va,outdir,device,a,seed):
    outdir.mkdir(parents=True,exist_ok=True);bp=outdir/'best_model.pth';hp=outdir/'training_history.csv';done=outdir/'FOLD_COMPLETE.json'
    if bp.exists() and hp.exists() and done.exists():
        info=json.load(open(done));print(f"      fold already complete -> best {info['best_val_mse']:.3f} @ {info['best_epoch']}");return bp,info['best_val_mse'],info['best_epoch'],info['runtime_seconds']
    torch.manual_seed(seed);np.random.seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    tl=make_loader(ds,tr,a.batch_size,True,device);vl=make_loader(ds,va,a.batch_size,False,device)
    model=InceptionV3_Flexible().to(device);model.model_3d_small=None;crit=nn.MSELoss();opt=optim.Adam(model.parameters(),lr=a.learning_rate);sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',patience=100)
    best=float('inf');be=0;hist=[];t0=time.time()
    for ep in range(1,a.epochs+1):
        model.train();ts=0;nb=0
        for b in tl:
            im,vo,de,y=batch_inputs(b,device);opt.zero_grad(set_to_none=True);pred=model(image=im,volume_small=None,volume_large=vo,descriptors=de);loss=crit(pred,y);loss.backward();opt.step();ts+=loss.item();nb+=1
        model.eval();vs=0;vn=0
        with torch.no_grad():
            for b in vl:
                im,vo,de,y=batch_inputs(b,device);loss=crit(model(image=im,volume_small=None,volume_large=vo,descriptors=de),y);vs+=loss.item();vn+=1
        vm=vs/max(vn,1);sch.step(vm);hist.append((ep,ts/max(nb,1),vm,opt.param_groups[0]['lr']))
        if vm<best:best=vm;be=ep;torch.save(model.state_dict(),bp)
        if ep==1 or ep%25==0 or ep==a.epochs:print(f'        epoch {ep:4d}/{a.epochs} val={vm:.3f} best={best:.3f}@{be}')
    runtime=time.time()-t0;pd.DataFrame(hist,columns=['epoch','train_mse','validation_mse','learning_rate']).to_csv(hp,index=False);json.dump({'best_val_mse':best,'best_epoch':be,'runtime_seconds':runtime},open(done,'w'),indent=2)
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    return bp,best,be,runtime

def ensemble_predict(ds,idx,paths,device,batch):
    loader=make_loader(ds,idx,batch,False,device);models=[]
    for p in paths:
        m=InceptionV3_Flexible().to(device);m.model_3d_small=None;m.load_state_dict(torch.load(p,map_location=device));m.eval();models.append(m)
    yy=[];pp=[]
    with torch.no_grad():
        for b in loader:
            im,vo,de,y=batch_inputs(b,device);pred=torch.stack([m(image=im,volume_small=None,volume_large=vo,descriptors=de) for m in models],0).mean(0);yy.extend(y.cpu().numpy().reshape(-1));pp.extend(pred.cpu().numpy().reshape(-1))
    del models
    if device.type=='cuda':torch.cuda.empty_cache()
    return np.asarray(yy,float),np.asarray(pp,float)

def main():
    a=parse_args();ddir=Path(a.dataset_dir);root=Path(a.results_dir);root.mkdir(parents=True,exist_ok=True);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('Device:',dev)
    meta,d4,aff,img,vol,y=build_common_cohort(ddir,load_arrays=True);ds=SignalDataset(d4,aff,img,vol,y);tests=[a.test] if a.test else list(range(1,11));aud=[]
    splitcache={}
    for test in tests:
        raw=pd.read_csv(ddir/f'test_{test}_updated.csv');codes={norm_code(v) for v in raw['Code']};codes.discard(None);testkeys=set(meta.loc[meta.Code.isin(codes),'StructureKey']);te=np.flatnonzero(meta.StructureKey.isin(testkeys).to_numpy());tr=np.flatnonzero((~meta.StructureKey.isin(testkeys)).to_numpy());overlap=set(meta.iloc[tr].StructureKey)&set(meta.iloc[te].StructureKey)
        aud.append({'test':test,'train_rows':len(tr),'test_rows':len(te),'train_structures':meta.iloc[tr].StructureKey.nunique(),'test_structures':meta.iloc[te].StructureKey.nunique(),'overlap':len(overlap)});splitcache[test]=(tr,te)
    adf=pd.DataFrame(aud);adf.to_csv(root/'split_audit.csv',index=False);print(adf.to_string(index=False));assert adf.overlap.max()==0
    if a.audit_only:return
    allm=[]
    for test in tests:
        tdir=root/f'test_{test:02d}';tdir.mkdir(parents=True,exist_ok=True);mp=tdir/'metrics.json';pp=tdir/'predictions_row_level.csv'
        if mp.exists() and pp.exists():print(f'TEST {test}: complete -> SKIP');allm.append(json.load(open(mp)));continue
        tr,te=splitcache[test];groups=meta.iloc[tr].StructureKey.to_numpy();gkf=GroupKFold(n_splits=5);paths=[];fs=[]
        print('\n'+'='*80+f'\nTEST {test}: train={len(tr)}, test={len(te)}, overlap=0\n'+'='*80)
        for fold,(aidx,vidx) in enumerate(gkf.split(np.zeros(len(tr)),y[tr],groups),1):
            ftr=tr[aidx];fva=tr[vidx];ov=set(meta.iloc[ftr].StructureKey)&set(meta.iloc[fva].StructureKey);assert not ov
            print(f'  fold {fold}/5 train={len(ftr)} val={len(fva)} group overlap=0')
            path,best,be,rt=train_fold(ds,ftr,fva,tdir/f'fold_{fold}',dev,a,RANDOM_STATE+test*1000+fold);paths.append(path);fs.append({'fold':fold,'best_val_mse':best,'best_epoch':be,'runtime_seconds':rt})
        true,pred=ensemble_predict(ds,te,paths,dev,a.batch_size);row,st,m=prediction_metrics(meta.iloc[te],true,pred);row.to_csv(pp,index=False);st.to_csv(tdir/'predictions_structure_level.csv',index=False);pd.DataFrame(fs).to_csv(tdir/'fold_summary.csv',index=False)
        m.update({'test':test,'train_rows':len(tr),'test_rows':len(te),'train_unique_structures':int(meta.iloc[tr].StructureKey.nunique()),'test_unique_structures':int(meta.iloc[te].StructureKey.nunique()),'train_test_structure_overlap':0})
        json.dump(m,open(mp,'w'),indent=2,allow_nan=True);allm.append(m);print(f"TEST {test}: structure MSE={m['structure_MSE']:.3f}, MAE={m['structure_MAE']:.3f}")
    df=pd.DataFrame(allm).sort_values('test');df.to_csv(root/'metrics_by_test.csv',index=False)
    if len(df)==10:
        s={'n_tests':10,'row_MAE_mean':float(df.row_MAE.mean()),'row_MAE_std':float(df.row_MAE.std(ddof=1)),'row_MSE_mean':float(df.row_MSE.mean()),'row_MSE_std':float(df.row_MSE.std(ddof=1)),'structure_MAE_mean':float(df.structure_MAE.mean()),'structure_MAE_std':float(df.structure_MAE.std(ddof=1)),'structure_MSE_mean':float(df.structure_MSE.mean()),'structure_MSE_std':float(df.structure_MSE.std(ddof=1)),'max_train_test_structure_overlap':int(df.train_test_structure_overlap.max())};json.dump(s,open(root/'metrics_summary.json','w'),indent=2);print('\nFINAL\n',json.dumps(s,indent=2))
if __name__=='__main__':main()
