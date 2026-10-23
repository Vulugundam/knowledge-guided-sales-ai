#!/usr/bin/env python3
"""Reproduce the knowledge-guided sales forecasting study in one command.

Creates exactly two manuscript tables and two publication figures from three
rolling 42-day tests, matched baselines, component ablations, and a paired
day-level statistical test. Customer count is excluded to prevent leakage.
"""
from __future__ import annotations
import argparse, gc, io, urllib.request
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_URL="https://raw.githubusercontent.com/Muosvr/rossmann-store-sales/master/data/train.csv"
SEED,HORIZON,FOLDS,TRAIN_DAYS=42,42,3,365

def load_data(path):
    if path: return pd.read_csv(path,low_memory=False)
    with urllib.request.urlopen(DATA_URL,timeout=180) as r: return pd.read_csv(io.BytesIO(r.read()),low_memory=False)

def prepare(df):
    cols=["Store","DayOfWeek","Date","Sales","Open","Promo","StateHoliday","SchoolHoliday"]
    missing=set(cols)-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    x=df[cols].copy(); x.Date=pd.to_datetime(x.Date); x.StateHoliday=x.StateHoliday.astype(str).ne("0").astype("int8")
    x["month"]=x.Date.dt.month; x["day"]=x.Date.dt.day; x["week"]=x.Date.dt.isocalendar().week.astype(int)
    x["trend"]=(x.Date-x.Date.min()).dt.days
    return x.sort_values(["Date","Store"]).reset_index(drop=True)

def wape(y,p): return float(np.abs(y-p).sum()/max(np.abs(y).sum(),1.0))
def rmspe(y,p):
    m=y>0; return float(np.sqrt(np.mean(((y[m]-p[m])/y[m])**2)))

def prior_maps(train,use_promo=True):
    op=train[(train.Open==1)&(train.Sales>0)]
    base=op[op.Promo==0].groupby(["Store","DayOfWeek"]).Sales.median(); fallback=op.groupby("Store").Sales.median(); glob=float(op.Sales.median())
    med=op.groupby(["Store","Promo"]).Sales.median().unstack(); uplift=(med.get(1)/med.get(0)).replace([np.inf,-np.inf],np.nan)
    uplift=uplift.clip(.8,2).fillna(1) if use_promo else pd.Series(1.,index=fallback.index)
    return base,fallback,glob,uplift

def make_prior(frame,maps,rules=True):
    base,fallback,glob,uplift=maps; key=pd.MultiIndex.from_frame(frame[["Store","DayOfWeek"]]); p=base.reindex(key).to_numpy(float)
    p=np.where(np.isnan(p),fallback.reindex(frame.Store).to_numpy(float),p); p=np.where(np.isnan(p),glob,p)
    p*=np.where(frame.Promo.to_numpy()==1,uplift.reindex(frame.Store).fillna(1).to_numpy(float),1)
    return np.where(frame.Open.to_numpy()==0,0,p) if rules else p

FEATURES=["Store","DayOfWeek","Open","Promo","StateHoliday","SchoolHoliday","month","day","week","trend","store_level"]
def matrix(x): return x[FEATURES].to_numpy(np.float32)

def fit_predict(train,test,use_prior,use_promo=True,rules=True):
    maps=prior_maps(train,use_promo); ptr=make_prior(train,maps,rules); pte=make_prior(test,maps,rules); xtr,xte=matrix(train),matrix(test)
    if use_prior: xtr=np.column_stack([xtr,np.log1p(ptr)]); xte=np.column_stack([xte,np.log1p(pte)])
    model=HistGradientBoostingRegressor(learning_rate=.09,max_iter=160,max_leaf_nodes=25,max_bins=127,l2_regularization=1,random_state=SEED)
    model.fit(xtr,np.log1p(train.Sales.to_numpy(float))); pred=np.maximum(np.expm1(model.predict(xte)),0)
    del model, xtr, xte; gc.collect()
    return (np.where(test.Open.to_numpy()==0,0,pred) if rules else pred),pte

def evaluate_fold(df,fold):
    end=df.Date.max()-pd.Timedelta(days=(FOLDS-1-fold)*HORIZON); start=end-pd.Timedelta(days=HORIZON-1)
    train_start=start-pd.Timedelta(days=TRAIN_DAYS)
    tr=df[(df.Date>=train_start)&(df.Date<start)].copy(); te=df[(df.Date>=start)&(df.Date<=end)].copy(); levels=tr.groupby("Store").Sales.median(); glob=float(tr.Sales.median())
    tr["store_level"]=tr.Store.map(levels).fillna(glob); te["store_level"]=te.Store.map(levels).fillna(glob)
    hybrid,prior=fit_predict(tr,te,True); data,_=fit_predict(tr,te,False); no_promo,_=fit_predict(tr,te,True,False); no_rules,_=fit_predict(tr,te,True,True,False)
    y=te.Sales.to_numpy(float); pred=pd.DataFrame({"Fold":fold+1,"Date":te.Date,"Sales":y,"Knowledge prior":prior,"Data-only AI":data,"Hybrid AI":hybrid,"Hybrid without promo knowledge":no_promo,"Hybrid without business rules":no_rules,"Promo":te.Promo,"SchoolHoliday":te.SchoolHoliday,"Open":te.Open})
    rows=[]
    for name in ["Knowledge prior","Data-only AI","Hybrid AI","Hybrid without promo knowledge","Hybrid without business rules"]:
        p=pred[name].to_numpy(); rows.append({"Fold":fold+1,"Model":name,"MAE":np.mean(np.abs(y-p)),"WAPE":wape(y,p),"RMSPE":rmspe(y,p)})
    return f"{start.date()} to {end.date()}",pred,pd.DataFrame(rows)

def table1(metrics,out):
    m=metrics[metrics.Model.isin(["Knowledge prior","Data-only AI","Hybrid AI"])]
    s=m.groupby("Model").agg(MAE_mean=("MAE","mean"),MAE_sd=("MAE","std"),WAPE_mean=("WAPE","mean"),WAPE_sd=("WAPE","std"),RMSPE_mean=("RMSPE","mean"),RMSPE_sd=("RMSPE","std")).reset_index()
    b=float(s.loc[s.Model.eq("Data-only AI"),"WAPE_mean"].iloc[0]); s["WAPE_gain_vs_data_only_pct"]=100*(b-s.WAPE_mean)/b; s.round(4).to_csv(out/"table1_model_performance.csv",index=False); return s

def table2(pred,metrics,out):
    rows=[]; base=metrics[metrics.Model.eq("Hybrid AI")].WAPE.mean()
    for name,label in [("Hybrid without promo knowledge","Remove promotion knowledge"),("Hybrid without business rules","Remove business rules")]:
        v=metrics[metrics.Model.eq(name)].WAPE.mean(); rows.append({"Diagnostic":label,"Hybrid_WAPE":base,"Comparator_WAPE":v,"Relative_gain_pct":100*(v-base)/v})
    for label,mask in [("Promotion days",pred.Promo.eq(1)),("Non-promotion days",pred.Promo.eq(0)),("School holidays",pred.SchoolHoliday.eq(1)),("All open days",pred.Open.eq(1))]:
        y=pred.loc[mask,"Sales"].to_numpy(); h=pred.loc[mask,"Hybrid AI"].to_numpy(); d=pred.loc[mask,"Data-only AI"].to_numpy(); hw,dw=wape(y,h),wape(y,d)
        rows.append({"Diagnostic":label,"Hybrid_WAPE":hw,"Comparator_WAPE":dw,"Relative_gain_pct":100*(dw-hw)/dw})
    t=pd.DataFrame(rows); t.round(4).to_csv(out/"table2_ablation_segments.csv",index=False); return t

def framework(path):
    fig,ax=plt.subplots(figsize=(11,4.2)); ax.axis("off"); boxes=[(.02,"Sales and\ncalendar data","#E8F1FA"),(.22,"Marketing\nknowledge prior","#D9EAD3"),(.43,"Hybrid gradient\nboosting model","#DCE6F1"),(.64,"Business-rule\nprojection","#FCE5CD"),(.83,"Decision-ready\nsales forecast","#EADCF8")]
    for x,text,color in boxes:
        ax.add_patch(FancyBboxPatch((x,.36),.15,.30,boxstyle="round,pad=.018",fc=color,ec="#23415A",lw=1.6)); ax.text(x+.075,.51,text,ha="center",va="center",fontsize=11,fontweight="bold",color="#172B3A")
    for a,b in zip([.17,.37,.58,.79],[.22,.43,.64,.83]): ax.add_patch(FancyArrowPatch((a,.51),(b,.51),arrowstyle="-|>",mutation_scale=16,lw=1.6,color="#315B7D"))
    ax.text(.295,.23,"store-weekday norm + bounded promotion uplift",ha="center",fontsize=9.5,color="#2F5E31"); ax.text(.715,.23,"non-negative sales + closed-store zero",ha="center",fontsize=9.5,color="#9A4E12")
    ax.text(.5,.89,"Knowledge-guided forecasting architecture",ha="center",fontsize=16,fontweight="bold",color="#102A43"); ax.text(.5,.06,"Explicit knowledge remains inspectable; AI learns nonlinear corrections around it.",ha="center",fontsize=10.5,color="#334E68")
    fig.savefig(path,dpi=400,bbox_inches="tight",facecolor="white"); plt.close(fig)

def results_figure(pred,metrics,t2,path):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9}); fig,axs=plt.subplots(2,2,figsize=(11,7.2)); orange="#D9792B"; green="#3A7D44"; navy="#24557A"
    pv=metrics[metrics.Model.isin(["Knowledge prior","Data-only AI","Hybrid AI"])].pivot(index="Fold",columns="Model",values="WAPE"); x=np.arange(FOLDS); w=.23
    for i,(name,color) in enumerate([("Knowledge prior","#8A9BA8"),("Data-only AI",orange),("Hybrid AI",green)]): axs[0,0].bar(x+(i-1)*w,pv[name]*100,w,label=name,color=color)
    axs[0,0].set_xticks(x,["Fold 1","Fold 2","Fold 3"]); axs[0,0].set_ylabel("WAPE (%)"); axs[0,0].set_title("A  Rolling-origin accuracy",loc="left",fontweight="bold"); axs[0,0].legend(frameon=False,fontsize=8)
    daily=pred.groupby("Date").apply(lambda g:pd.Series({"Hybrid":wape(g.Sales.values,g["Hybrid AI"].values),"Data":wape(g.Sales.values,g["Data-only AI"].values)}),include_groups=False); delta=(daily.Data-daily.Hybrid)*100
    axs[0,1].hist(delta,bins=24,color=navy,alpha=.88,edgecolor="white"); axs[0,1].axvline(0,color="#A61B1B",ls="--",lw=1.4); axs[0,1].set_xlabel("Daily WAPE improvement (percentage points)"); axs[0,1].set_title("B  Paired daily improvement",loc="left",fontweight="bold")
    seg=t2.iloc[2:]; sx=np.arange(len(seg)); axs[1,0].bar(sx-.17,seg.Hybrid_WAPE*100,.34,label="Hybrid AI",color=green); axs[1,0].bar(sx+.17,seg.Comparator_WAPE*100,.34,label="Data-only AI",color=orange)
    axs[1,0].set_xticks(sx,["Promo","No promo","School\nholiday","Open"]); axs[1,0].set_ylabel("WAPE (%)"); axs[1,0].set_title("C  Marketing segments",loc="left",fontweight="bold"); axs[1,0].legend(frameon=False,fontsize=8)
    last=pred[pred.Fold.eq(FOLDS)].groupby("Date")[["Sales","Hybrid AI","Data-only AI"]].sum()/1e6
    axs[1,1].plot(last.index,last.Sales,color="#222",lw=2,label="Actual"); axs[1,1].plot(last.index,last["Hybrid AI"],color=green,lw=1.7,label="Hybrid AI"); axs[1,1].plot(last.index,last["Data-only AI"],color=orange,lw=1.2,label="Data-only AI")
    axs[1,1].set_ylabel("Daily sales (millions)"); axs[1,1].set_title("D  Final-fold sales trajectory",loc="left",fontweight="bold"); axs[1,1].legend(frameon=False,fontsize=8); axs[1,1].tick_params(axis="x",rotation=25)
    for ax in axs.flat: ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.18)
    fig.suptitle("Robustness and marketing-performance diagnostics",fontsize=15,fontweight="bold",color="#102A43",y=.995); fig.tight_layout(); fig.savefig(path,dpi=400,bbox_inches="tight",facecolor="white"); plt.close(fig)
    stat,p=wilcoxon(daily.Data,daily.Hybrid,alternative="greater"); return len(daily),float(stat),float(p),float((daily.Hybrid<daily.Data).mean())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--output",default="results"); a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); df=prepare(load_data(a.data))
    folds=[evaluate_fold(df,i) for i in range(FOLDS)]; pred=pd.concat([x[1] for x in folds],ignore_index=True); metrics=pd.concat([x[2] for x in folds],ignore_index=True); t1=table1(metrics,out); t2=table2(pred,metrics,out); framework(out/"figure1_framework.png"); n,stat,p,win=results_figure(pred,metrics,t2,out/"figure2_results.png")
    print("Rolling windows:"); [print(f"  Fold {i+1}: {x[0]}") for i,x in enumerate(folds)]; print("\nTable 1\n",t1.round(4).to_string(index=False)); print("\nTable 2\n",t2.round(4).to_string(index=False)); print(f"\nPaired daily Wilcoxon: n={n}, statistic={stat:.1f}, p={p:.3g}; hybrid win rate={win:.1%}")

if __name__=="__main__": main()
