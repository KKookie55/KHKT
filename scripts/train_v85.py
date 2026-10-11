#!/usr/bin/env python3
"""v8.5 TRAINING — 5 models × 4 calibrators × 5 splits × 3 horizons × N seeds.
Resumable. Supports --label_source modis|viirs."""
import os, sys, json, time, math, warnings, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import xgboost as xgb
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              brier_score_loss, precision_recall_curve)
from sklearn.isotonic import IsotonicRegression

OUT = Path("/home/z/my-project/download/real_data_v85")
RESULTS_DIR = OUT / "runs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_PHYS_COLS = ["t2m","rh","wind"]

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def metrics(y, p, f1t=0.5):
    y=np.asarray(y); p=np.asarray(p)
    try: a=roc_auc_score(y,p)
    except: a=0.5
    try: ap=average_precision_score(y,p)
    except: ap=0.0
    b=brier_score_loss(y,p)
    bins=np.linspace(0,1,11); e=0; n=len(y)
    for i in range(10):
        lo,hi=bins[i],bins[i+1]
        ib=((p>=lo)&(p<=hi)) if i==9 else ((p>=lo)&(p<hi))
        if ib.sum()==0: continue
        e+=(ib.sum()/n)*abs(y[ib].mean()-p[ib].mean())
    return {"AUROC":float(a),"AUPRC":float(ap),"F1":float(f1_score(y,(p>=f1t).astype(int),zero_division=0)),
            "Brier":float(b),"ECE":float(e)}

def calibrate_temperature(p_val, y_val, p_test):
    eps=1e-6
    lv=np.log(np.clip(p_val,eps,1-eps)/np.clip(1-p_val,eps,1-eps))
    lt=np.log(np.clip(p_test,eps,1-eps)/np.clip(1-p_test,eps,1-eps))
    best_T,best_nll=1.0,np.inf
    for T in np.linspace(0.5,5.0,31):
        scaled=1/(1+np.exp(-lv/T))
        nll=-np.mean(y_val*np.log(scaled+eps)+(1-y_val)*np.log(1-scaled+eps))
        if nll<best_nll: best_nll=nll; best_T=T
    return 1/(1+np.exp(-lt/best_T))

def calibrate_isotonic(p_val, y_val, p_test):
    iso=IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(p_val,y_val); return iso.predict(p_test)

def calibrate_platt(p_val, y_val, p_test):
    eps=1e-6
    lv=np.log(np.clip(p_val,eps,1-eps)/np.clip(1-p_val,eps,1-eps))
    lt=np.log(np.clip(p_test,eps,1-eps)/np.clip(1-p_test,eps,1-eps))
    lr=LogisticRegression(C=1e10, solver="lbfgs", max_iter=200)
    lr.fit(lv.reshape(-1,1), y_val)
    return lr.predict_proba(lt.reshape(-1,1))[:,1]

def train_xgb(X_tr,y_tr,X_val,y_val,seed=42):
    pos=max(int(y_tr.sum()),1); neg=max(int((1-y_tr).sum()),1)
    spw=max(math.sqrt(neg/pos),1.0)
    m=xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.85,
        colsample_bytree=0.7,min_child_weight=20,reg_lambda=3.0,reg_alpha=0.5,gamma=0.1,
        scale_pos_weight=spw,eval_metric="aucpr",tree_method="hist",random_state=seed,n_jobs=4)
    m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
    return m, lambda X: m.predict_proba(X)[:,1]

def train_rf(X_tr,y_tr,seed=42):
    imp=SimpleImputer(strategy="median"); X=imp.fit_transform(X_tr)
    m=RandomForestClassifier(n_estimators=50,max_depth=6,min_samples_leaf=20,
        class_weight="balanced",random_state=seed,n_jobs=4)
    m.fit(X,y_tr)
    return m, lambda X: m.predict_proba(imp.transform(X))[:,1]

def train_logreg(X_tr,y_tr,seed=42):
    sc=StandardScaler(); imp=SimpleImputer(strategy="median")
    X_imp=imp.fit_transform(X_tr); X_scaled=sc.fit_transform(X_imp)
    m=LogisticRegression(C=1.0,class_weight="balanced",max_iter=500,random_state=seed)
    m.fit(X_scaled,y_tr)
    return m, lambda X: m.predict_proba(sc.transform(imp.transform(X)))[:,1]

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.bn1=nn.BatchNorm1d(dim); self.fc1=nn.Linear(dim,dim)
        self.bn2=nn.BatchNorm1d(dim); self.fc2=nn.Linear(dim,dim); self.drop=nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.fc1.weight,nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.fc2.weight,nonlinearity="leaky_relu")
        nn.init.zeros_(self.fc1.bias); nn.init.zeros_(self.fc2.bias)
    def forward(self,x):
        out=F.silu(self.fc1(self.bn1(x))); out=self.drop(out); out=F.silu(self.fc2(self.bn2(out)))
        return x+out

class PhysicsMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.15):
        super().__init__()
        self.input_proj=nn.Linear(in_dim,hidden)
        nn.init.kaiming_normal_(self.input_proj.weight,nonlinearity="leaky_relu"); nn.init.zeros_(self.input_proj.bias)
        self.res=ResBlock(hidden,dropout); self.out_bn=nn.BatchNorm1d(hidden); self.out_fc=nn.Linear(hidden,1)
        nn.init.normal_(self.out_fc.weight,std=0.01); nn.init.zeros_(self.out_fc.bias)
    def forward(self,x):
        h=F.silu(self.input_proj(x)); h=self.res(h)
        return self.out_fc(self.out_bn(h)).squeeze(-1)

def physics_penalty_raw(pred, raw_batch):
    p=torch.sigmoid(pred); t=raw_batch[:,0]; rh=raw_batch[:,1]; w=raw_batch[:,2]
    n=p.shape[0]
    if n<8: return torch.tensor(0.0, device=pred.device)
    sz=n//2; ii=torch.randint(0,n,(sz,),device=pred.device); ij=torch.randint(0,n,(sz,),device=pred.device)
    td=t[ii]-t[ij]; pd=p[ii]-p[ij]
    l_e=torch.mean((td*pd<0).float()*torch.abs(pd))
    rd=rh[ii]-rh[ij]; l_m=torch.mean((rd*pd>0).float()*torch.abs(pd))
    hm=(t[ii]>25)&(t[ij]>25); wd=w[ii]-w[ij]
    l_w=torch.sum(hm.float()*(wd*pd<0).float()*torch.abs(pd))/(hm.sum().float()+1e-6)
    return 0.4*l_e+0.4*l_m+0.2*l_w

def train_mlp(X_tr,y_tr,X_val,y_val,X_raw_tr,seed=42,lam=0.0,epochs=15,batch_size=1024,lr=1e-3):
    set_seed(seed)
    imp=SimpleImputer(strategy="median"); X_tr_imp=imp.fit_transform(X_tr)
    scaler=StandardScaler(); X_tr_std=scaler.fit_transform(X_tr_imp)
    X_val_std=scaler.transform(imp.transform(X_val))
    X_t=torch.tensor(X_tr_std,dtype=torch.float32); y_t=torch.tensor(np.asarray(y_tr),dtype=torch.float32)
    X_r=torch.tensor(np.nan_to_num(X_raw_tr.astype(np.float32)))
    model=PhysicsMLP(X_t.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    bce=nn.BCEWithLogitsLoss()
    ds=TensorDataset(X_t,y_t,X_r); dl=DataLoader(ds,batch_size=batch_size,shuffle=True)
    phase1=int(epochs*0.5)
    for ep in range(epochs):
        model.train(); use_phys=ep>=phase1
        for xb,yb,xr in dl:
            opt.zero_grad(); logits=model(xb); loss=bce(logits,yb)
            if use_phys and lam>0: loss=loss+lam*physics_penalty_raw(logits,xr)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0); opt.step()
        sched.step()
    def predict_fn(X):
        X_imp=imp.transform(X); X_std=scaler.transform(X_imp)
        model.eval()
        with torch.no_grad(): return torch.sigmoid(model(torch.tensor(X_std,dtype=torch.float32))).numpy()
    return model, predict_fn

def run_one(split_name, horizon, seed, model_name, panel, splits, features, label_source="modis"):
    out_file = RESULTS_DIR / f"{label_source}__{split_name}__h{horizon}__s{seed}__{model_name}.json"
    if out_file.exists(): return json.loads(out_file.read_text())
    t0=time.time(); set_seed(seed)
    label_col = f"label_h{horizon}"
    sp = splits[split_name]
    tr = panel.iloc[np.array(sp["train_idx"])].dropna(subset=[label_col])
    va = panel.iloc[np.array(sp["val_idx"])].dropna(subset=[label_col])
    te = panel.iloc[np.array(sp["test_idx"])].dropna(subset=[label_col])
    X_tr=tr[features].values.astype(np.float32); y_tr=tr[label_col].values.astype(np.float32)
    X_val=va[features].values.astype(np.float32); y_val=va[label_col].values.astype(np.float32)
    X_test=te[features].values.astype(np.float32); y_test=te[label_col].values.astype(np.float32)
    if "precip_roll30_sum" in features:
        di_idx=features.index("precip_roll30_sum")
        di_mean=np.nanmean(X_tr[:,di_idx]); di_std=np.nanstd(X_tr[:,di_idx])+1e-6
        for arr in [X_tr,X_val,X_test]:
            arr[:,di_idx]=np.nan_to_num((arr[:,di_idx]-di_mean)/di_std, nan=0.0)
    X_raw_tr=tr[RAW_PHYS_COLS].values.astype(np.float32)
    try:
        if model_name=="LogReg": _,pf=train_logreg(X_tr,y_tr,seed=seed)
        elif model_name=="RF": _,pf=train_rf(X_tr,y_tr,seed=seed)
        elif model_name=="XGB": _,pf=train_xgb(X_tr,y_tr,X_val,y_val,seed=seed)
        elif model_name=="MLP": _,pf=train_mlp(X_tr,y_tr,X_val,y_val,X_raw_tr,seed=seed,lam=0.0)
        elif model_name=="PhysMLP": _,pf=train_mlp(X_tr,y_tr,X_val,y_val,X_raw_tr,seed=seed,lam=0.10)
        else: raise ValueError(model_name)
    except Exception as e:
        print(f"    ERROR {model_name}: {e}"); return None
    p_val=pf(X_val); p_test=pf(X_test)
    def f1t(y_v,p_v):
        prec,rec,thr=precision_recall_curve(y_v,p_v)
        f1s=2*prec*rec/(prec+rec+1e-12)
        return float(thr[int(np.argmax(f1s[:-1]))]) if len(thr)>0 else 0.5
    results={}
    f1t_raw=f1t(y_val,p_val); results["Raw"]=metrics(y_test,p_test,f1t_raw)
    p_val_t=calibrate_temperature(p_val,y_val,p_val); p_test_t=calibrate_temperature(p_val,y_val,p_test)
    results["Temp"]=metrics(y_test,p_test_t,f1t(y_val,p_val_t))
    p_val_i=calibrate_isotonic(p_val,y_val,p_val); p_test_i=calibrate_isotonic(p_val,y_val,p_test)
    results["Iso"]=metrics(y_test,p_test_i,f1t(y_val,p_val_i))
    p_val_p=calibrate_platt(p_val,y_val,p_val); p_test_p=calibrate_platt(p_val,y_val,p_test)
    results["Platt"]=metrics(y_test,p_test_p,f1t(y_val,p_val_p))
    out={"split":split_name,"horizon":horizon,"seed":seed,"model":model_name,"label_source":label_source,
         "n_train":len(tr),"n_val":len(va),"n_test":len(te),
         "n_pos_train":int(y_tr.sum()),"n_pos_val":int(y_val.sum()),"n_pos_test":int(y_test.sum()),
         "f1_threshold_raw":f1t_raw,"metrics":results,"wall_time_s":time.time()-t0}
    out_file.write_text(json.dumps(out,indent=2,default=str))
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--seeds",type=int,nargs="+",default=[42])
    p.add_argument("--horizons",type=int,nargs="+",default=[1])
    p.add_argument("--splits",type=str,nargs="+",default=["all"])
    p.add_argument("--models",type=str,nargs="+",default=["all"])
    p.add_argument("--label_source",type=str,default="modis",choices=["modis","viirs"])
    args=p.parse_args()
    if args.smoke:
        seeds=[42]; horizons=[1]
        all_splits=["IID_groupblocked","Temporal_OOD","Spatial_OOD_NorthToSouth",
                    "Spatial_OOD_SouthToCentral","Spatial_OOD_CentralSouthToNorth"]
        all_models=["LogReg","RF","XGB","MLP","PhysMLP"]
    else:
        seeds=args.seeds; horizons=args.horizons
        all_splits=(["IID_groupblocked","Temporal_OOD","Spatial_OOD_NorthToSouth",
                     "Spatial_OOD_SouthToCentral","Spatial_OOD_CentralSouthToNorth"]
                    if args.splits==["all"] else args.splits)
        all_models=(["LogReg","RF","XGB","MLP","PhysMLP"] if args.models==["all"] else args.models)
    print("="*70); print(f"v8.5 TRAINING (label_source={args.label_source})")
    print(f"  Splits:{all_splits}\n  Horizons:{horizons}\n  Seeds:{seeds}\n  Models:{all_models}")
    print(f"  Total: {len(all_splits)*len(horizons)*len(seeds)*len(all_models)} runs"); print("="*70)
    panel_file = OUT/("panel_v85.parquet" if args.label_source=="modis" else "panel_v85_viirs.parquet")
    panel = pd.read_parquet(panel_file)
    panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
    splits_npz = np.load(OUT/"splits_v85.npz")
    splits = {sn: {"train_idx": splits_npz[f"{sn}_train"].tolist(),
                  "val_idx": splits_npz[f"{sn}_val"].tolist(),
                  "test_idx": splits_npz[f"{sn}_test"].tolist()} for sn in all_splits}
    features = json.loads((OUT/"panel_v85_meta.json").read_text())["features"]
    print(f"  Panel: {len(panel):,} rows, {len(features)} features")
    total = len(all_splits)*len(horizons)*len(seeds)*len(all_models); i=0; t0=time.time()
    for sn in all_splits:
        for h in horizons:
            for s in seeds:
                for mn in all_models:
                    i+=1
                    el=time.time()-t0; eta=(el/max(i-1,1))*(total-i+1) if i>1 else 0
                    print(f"\n[{i}/{total}] {args.label_source} {sn} h{h} s{s} {mn} (el {el:.0f}s, ETA {eta:.0f}s)")
                    r=run_one(sn,h,s,mn,panel,splits,features,label_source=args.label_source)
                    if r:
                        for c,m in r["metrics"].items():
                            print(f"    {c:6s}: AUROC={m['AUROC']:.4f} ECE={m['ECE']:.4f} F1={m['F1']:.4f}")
    print(f"\n{'='*70}\nAGGREGATING...\n{'='*70}")
    aggregate(all_splits,horizons,seeds,all_models,args.label_source)

def aggregate(all_splits,horizons,seeds,all_models,ls="modis"):
    from collections import defaultdict
    all_runs=[]
    for f in RESULTS_DIR.glob(f"{ls}__*.json"):
        try: all_runs.append(json.loads(f.read_text()))
        except: pass
    if not all_runs: print(f"  No runs for {ls}."); return
    grouped=defaultdict(list)
    for r in all_runs:
        for c,m in r["metrics"].items():
            grouped[(r["split"],r["horizon"],r["model"],c)].append(m)
    summary=[]
    for (sp,h,m,c),runs in sorted(grouped.items()):
        row={"split":sp,"horizon":h,"model":m,"calibrator":c,"n_seeds":len(runs),"label_source":ls}
        for met in ["AUROC","AUPRC","F1","Brier","ECE"]:
            vals=[r[met] for r in runs]
            row[f"{met}_mean"]=float(np.mean(vals)); row[f"{met}_std"]=float(np.std(vals))
        summary.append(row)
    sdf=pd.DataFrame(summary)
    sdf.to_csv(OUT/f"results_v85_summary_{ls}.csv",index=False)
    print(f"  Saved: results_v85_summary_{ls}.csv ({len(all_runs)} runs, {len(summary)} groups)")
    print(f"\n  Focus ({ls}, IID vs Spatial N→S, t+1, Iso):")
    focus = sdf[(sdf.horizon==1)&(sdf.split.isin(["IID_groupblocked","Spatial_OOD_NorthToSouth"]))&
                (sdf.model.isin(["XGB","MLP","PhysMLP"]))&(sdf.calibrator.isin(["Raw","Iso"]))]
    if len(focus):
        print(focus[["split","model","calibrator","AUROC_mean","AUROC_std","ECE_mean","ECE_std","Brier_mean"]].to_string(index=False))

if __name__=="__main__": main()
