# =============================================================================
# EV Battery Health Prediction — State of Health & RUL Estimation
# Client: BMW Group — Electric Vehicle Division
# Dataset: NASA Ames Li-ion Battery Aging Dataset (PCoE)
# =============================================================================
import warnings
warnings.filterwarnings("ignore")
import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.io import loadmat
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
    r2_score, accuracy_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi":130,"axes.titlesize":13})
ROOT=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(ROOT,"data"); OUTPUTS=os.path.join(ROOT,"outputs")
os.makedirs(OUTPUTS,exist_ok=True)

def section(t): print("\n"+"="*65+f"\n  {t}\n"+"="*65)
def save(fig,n): fig.savefig(os.path.join(OUTPUTS,n),bbox_inches="tight"); print(f"    ✔ Saved: {n}")

BATTERIES=["B0005","B0006","B0007","B0018"]
RATED_CAP=2.0; EOL_CAP=1.4; EOL_SOH=EOL_CAP/RATED_CAP

def extract_battery_data(bid):
    mat=loadmat(os.path.join(DATA,f"{bid}.mat"))
    data=mat[bid][0,0]["cycle"][0]
    records=[]; cycle_num=0
    for i in range(len(data)):
        row=data[i]; ctype=str(row["type"][0])
        if ctype=="discharge":
            cycle_num+=1
            temp=float(row["ambient_temperature"][0][0])
            meas=row["data"][0,0]
            voltage=meas["Voltage_measured"][0]
            current=meas["Current_measured"][0]
            temperature=meas["Temperature_measured"][0]
            capacity=float(meas["Capacity"][0][0])
            records.append({"battery_id":bid,"cycle":cycle_num,
                "capacity_ah":capacity,"ambient_temp_c":temp,
                "voltage_mean":np.mean(voltage),"voltage_min":np.min(voltage),
                "voltage_std":np.std(voltage),
                "voltage_range":np.max(voltage)-np.min(voltage),
                "current_mean":np.mean(np.abs(current)),"current_std":np.std(current),
                "temp_mean":np.mean(temperature),"temp_max":np.max(temperature),
                "temp_rise":np.max(temperature)-np.min(temperature),
                "discharge_duration":len(voltage)})
    df=pd.DataFrame(records)
    df["SoH"]=df["capacity_ah"]/RATED_CAP
    eol_cycles=df[df["SoH"]<=EOL_SOH]["cycle"]
    eol_cycle=int(eol_cycles.min()) if len(eol_cycles)>0 else int(df["cycle"].max())
    df["RUL"]=(eol_cycle-df["cycle"]).clip(lower=0)
    df["condition"]=pd.cut(df["SoH"],bins=[0,0.75,0.88,1.1],
        labels=["Near EOL","Degraded","Healthy"])
    print(f"  {bid}: {len(df)} cycles | EOL cycle {eol_cycle} | Final SoH {df['SoH'].iloc[-1]:.3f}")
    return df

section("STAGE 1 — LOADING NASA BATTERY DATASET")
df=pd.concat([extract_battery_data(b) for b in BATTERIES],ignore_index=True)
print(f"\n  Total cycles: {len(df):,} | Batteries: {df['battery_id'].nunique()}")
print(f"  SoH range: {df['SoH'].min():.3f} to {df['SoH'].max():.3f}")
print(f"  Condition split:\n{df['condition'].value_counts().to_string()}")

section("STAGE 2 — EDA")
colors=["#00B89F","#E74C3C","#3498DB","#F39C12"]
fig,axes=plt.subplots(1,2,figsize=(15,5))
for i,b in enumerate(BATTERIES):
    bdf=df[df["battery_id"]==b]
    axes[0].plot(bdf["cycle"],bdf["SoH"],color=colors[i],linewidth=2,label=b)
axes[0].axhline(EOL_SOH,color="red",linestyle="--",lw=1.5,label=f"EOL (SoH={EOL_SOH})")
axes[0].set_title("Battery State of Health — All 4 Batteries",fontweight="bold")
axes[0].set_xlabel("Cycle"); axes[0].set_ylabel("State of Health (SoH)"); axes[0].legend()
cc=df["condition"].value_counts()
axes[1].pie(cc.values,labels=cc.index,colors=["#27AE60","#F39C12","#E74C3C"],
    autopct="%1.1f%%",startangle=90,wedgeprops=dict(edgecolor="white",linewidth=2))
axes[1].set_title("Battery Condition Distribution",fontweight="bold")
plt.suptitle("NASA Li-ion Battery Dataset Overview",fontsize=14,fontweight="bold")
plt.tight_layout(); save(fig,"eda_01_degradation_overview.png"); plt.show()

fig,axes=plt.subplots(2,3,figsize=(16,10)); axes=axes.flatten()
pairs=[("cycle","SoH","Cycle vs SoH"),("voltage_mean","SoH","Mean Voltage vs SoH"),
    ("temp_rise","SoH","Temp Rise vs SoH"),("discharge_duration","SoH","Duration vs SoH"),
    ("voltage_std","SoH","Voltage Std vs SoH"),("temp_max","SoH","Max Temp vs SoH")]
for i,(x,y,t) in enumerate(pairs):
    for j,b in enumerate(BATTERIES):
        bdf=df[df["battery_id"]==b]
        axes[i].scatter(bdf[x],bdf[y],alpha=0.4,s=8,color=colors[j])
    axes[i].set_xlabel(x); axes[i].set_ylabel(y); axes[i].set_title(t,fontweight="bold")
plt.suptitle("Feature Correlations with SoH",fontsize=14,fontweight="bold")
plt.tight_layout(); save(fig,"eda_02_feature_correlations.png"); plt.show()

section("STAGE 3 — FEATURE ENGINEERING")
FEATURE_COLS=["cycle","ambient_temp_c","voltage_mean","voltage_min","voltage_std",
    "voltage_range","current_mean","current_std","temp_mean","temp_max","temp_rise",
    "discharge_duration"]
WINDOW=5; df=df.sort_values(["battery_id","cycle"]).reset_index(drop=True)
for feat in ["voltage_mean","temp_rise","discharge_duration","capacity_ah"]:
    grp=df.groupby("battery_id")[feat]
    df[f"{feat}_roll_mean"]=grp.transform(lambda x:x.rolling(WINDOW,min_periods=1).mean())
    df[f"{feat}_roll_std"]=grp.transform(lambda x:x.rolling(WINDOW,min_periods=1).std().fillna(0))
df["cap_fade_rate"]=df.groupby("battery_id")["capacity_ah"].transform(lambda x:x.diff().fillna(0))
df["cycle_norm"]=df.groupby("battery_id")["cycle"].transform(
    lambda x:(x-x.min())/(x.max()-x.min()+1e-8))
ALL_FEATURES=FEATURE_COLS+["voltage_mean_roll_mean","voltage_mean_roll_std",
    "temp_rise_roll_mean","temp_rise_roll_std","discharge_duration_roll_mean",
    "capacity_ah_roll_mean","cap_fade_rate","cycle_norm"]
print(f"  Total features: {len(ALL_FEATURES)}")

section("STAGE 4 — MODELLING")
TRAIN_BATS=["B0005","B0006","B0007"]; TEST_BAT="B0018"
tr=df[df["battery_id"].isin(TRAIN_BATS)].dropna(subset=ALL_FEATURES)
te=df[df["battery_id"]==TEST_BAT].dropna(subset=ALL_FEATURES)
X_tr=tr[ALL_FEATURES]; X_te=te[ALL_FEATURES]
print(f"  Train: {len(X_tr):,} cycles | Test: {len(X_te):,} cycles")

# SoH model
soh_model=RandomForestRegressor(n_estimators=300,max_depth=12,min_samples_leaf=3,random_state=42,n_jobs=-1)
soh_model.fit(X_tr,tr["SoH"]); soh_pred=soh_model.predict(X_te)
soh_rmse=np.sqrt(mean_squared_error(te["SoH"],soh_pred))
soh_mae=mean_absolute_error(te["SoH"],soh_pred)
soh_r2=r2_score(te["SoH"],soh_pred)
print(f"\n  SoH Model: RMSE={soh_rmse:.4f} ({soh_rmse*100:.2f}%) | MAE={soh_mae:.4f} | R2={soh_r2:.4f}")

# RUL model
rul_model=RandomForestRegressor(n_estimators=300,max_depth=12,min_samples_leaf=3,random_state=42,n_jobs=-1)
rul_model.fit(X_tr,tr["RUL"]); rul_pred=rul_model.predict(X_te)
rul_rmse=np.sqrt(mean_squared_error(te["RUL"],rul_pred))
rul_mae=mean_absolute_error(te["RUL"],rul_pred)
rul_r2=r2_score(te["RUL"],rul_pred)
print(f"  RUL Model: RMSE={rul_rmse:.2f} cycles | MAE={rul_mae:.2f} cycles | R2={rul_r2:.4f}")

# Condition model
le=LabelEncoder()
yc_tr=le.fit_transform(tr["condition"].astype(str))
yc_te=le.transform(te["condition"].astype(str))
cond_model=RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42,n_jobs=-1)
cond_model.fit(X_tr,yc_tr); cond_pred=cond_model.predict(X_te)
cond_acc=accuracy_score(yc_te,cond_pred)
print(f"  Condition Model: Accuracy={cond_acc:.4f}")
print(f"\n{classification_report(yc_te,cond_pred,target_names=le.classes_)}")

section("STAGE 5 — EVALUATION PLOTS")
fig=plt.figure(figsize=(18,12)); gs=gridspec.GridSpec(2,3,figure=fig)
ax1=fig.add_subplot(gs[0,0])
ax1.scatter(te["SoH"],soh_pred,alpha=0.6,color="#00B89F",s=20,edgecolors="white",linewidth=0.3)
mn=min(te["SoH"].min(),soh_pred.min()); mx=max(te["SoH"].max(),soh_pred.max())
ax1.plot([mn,mx],[mn,mx],"r--",lw=1.5,label="Perfect")
ax1.set_xlabel("True SoH"); ax1.set_ylabel("Predicted SoH")
ax1.set_title(f"SoH Regression\nRMSE={soh_rmse:.4f} R2={soh_r2:.3f}",fontweight="bold"); ax1.legend()

ax2=fig.add_subplot(gs[0,1])
ax2.plot(te["cycle"].values,te["SoH"].values,color="#1C1C1C",linewidth=2,label="True SoH")
ax2.plot(te["cycle"].values,soh_pred,color="#00B89F",linewidth=2,linestyle="--",label="Predicted SoH")
ax2.axhline(EOL_SOH,color="red",linestyle=":",lw=1.5,label="EOL")
ax2.set_title("SoH Over Cycles — Battery B0018",fontweight="bold")
ax2.set_xlabel("Cycle"); ax2.set_ylabel("SoH"); ax2.legend(fontsize=9)

ax3=fig.add_subplot(gs[0,2])
ax3.scatter(te["RUL"],rul_pred,alpha=0.6,color="#3498DB",s=20,edgecolors="white",linewidth=0.3)
mn=min(te["RUL"].min(),rul_pred.min()); mx=max(te["RUL"].max(),rul_pred.max())
ax3.plot([mn,mx],[mn,mx],"r--",lw=1.5,label="Perfect")
ax3.set_xlabel("True RUL"); ax3.set_ylabel("Predicted RUL")
ax3.set_title(f"RUL Regression\nRMSE={rul_rmse:.1f} R2={rul_r2:.3f}",fontweight="bold"); ax3.legend()

ax4=fig.add_subplot(gs[1,0])
cm=confusion_matrix(yc_te,cond_pred)
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax4,
    xticklabels=le.classes_,yticklabels=le.classes_,
    linewidths=1,annot_kws={"size":11})
ax4.set_title(f"Condition Classification\nAccuracy={cond_acc:.3f}",fontweight="bold")
ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")

ax5=fig.add_subplot(gs[1,1])
fi=pd.Series(soh_model.feature_importances_,index=ALL_FEATURES).nlargest(10).sort_values()
ax5.barh(fi.index,fi.values,color=plt.cm.RdYlGn(np.linspace(0.2,0.9,len(fi))),edgecolor="white")
ax5.set_title("Top 10 Features — SoH Model",fontweight="bold"); ax5.set_xlabel("Importance")

ax6=fig.add_subplot(gs[1,2])
ax6.plot(te["cycle"].values,te["RUL"].values,color="#1C1C1C",linewidth=2,label="True RUL")
ax6.plot(te["cycle"].values,rul_pred,color="#E74C3C",linewidth=2,linestyle="--",label="Predicted RUL")
ax6.axhline(30,color="orange",linestyle=":",lw=1.5,label="Alert zone")
ax6.set_title("RUL Prediction — B0018",fontweight="bold")
ax6.set_xlabel("Cycle"); ax6.set_ylabel("RUL (cycles)"); ax6.legend(fontsize=9)

plt.suptitle("EV Battery Health — Model Evaluation Dashboard | BMW Group | NASA Dataset",
    fontsize=13,fontweight="bold")
plt.tight_layout(); save(fig,"model_01_evaluation_dashboard.png"); plt.show()

section("STAGE 6 — SAVING RESULTS")
out=te[["battery_id","cycle","capacity_ah","SoH","RUL","condition"]].copy()
out["predicted_SoH"]=soh_pred.round(4)
out["predicted_RUL"]=rul_pred.round(0).astype(int)
out["predicted_condition"]=le.inverse_transform(cond_pred)
out.to_csv(os.path.join(OUTPUTS,"battery_predictions.csv"),index=False)

report=f"""
=================================================================
  BMW GROUP ELECTRIC VEHICLE DIVISION
  Battery Health Prediction Report
  Dataset: NASA Ames Li-ion Battery Aging (PCoE)
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
=================================================================

MODEL 1 - State of Health (SoH) Regression:
  RMSE : {soh_rmse:.4f} ({soh_rmse*100:.2f}% average error)
  MAE  : {soh_mae:.4f} ({soh_mae*100:.2f}% average error)
  R2   : {soh_r2:.4f}

MODEL 2 - Remaining Useful Life (RUL) Regression:
  RMSE : {rul_rmse:.2f} cycles
  MAE  : {rul_mae:.2f} cycles
  R2   : {rul_r2:.4f}

MODEL 3 - Condition Classification:
  Accuracy : {cond_acc:.4f}
  Classes  : Healthy / Degraded / Near EOL

BUSINESS IMPACT:
  -> Accurate SoH tells drivers exact remaining range
  -> RUL prediction enables proactive replacement planning
  -> Condition alerts prevent unexpected battery failures
  -> Reduces unnecessary early battery replacements
  -> Supports BMW warranty cost optimisation

RECOMMENDATIONS:
  1. Deploy SoH model in BMW Battery Management System
  2. Alert driver when SoH drops below 80%
  3. Schedule replacement when RUL < 30 cycles
  4. Retrain monthly with new fleet battery data
  5. Extend to BMW iX, i4, i7 full battery datasets
=================================================================
"""
print(report)
with open(os.path.join(OUTPUTS,"executive_summary.txt"),"w",encoding="utf-8") as f:
    f.write(report)
section("PIPELINE COMPLETE — ALL OUTPUTS SAVED")
