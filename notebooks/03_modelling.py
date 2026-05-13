# =============================================================================
# EV Battery Health Prediction — Notebook 3: Modelling & Evaluation
# Client: BMW Group — Electric Vehicle Division
# Dataset: NASA Ames Li-ion Battery Aging Dataset (PCoE)
# =============================================================================
# Goal: Train and evaluate 3 models to predict battery health.
#
# MODEL 1 — SoH Regression
#   Predict exact State of Health (%) for real-time dashboard display
#
# MODEL 2 — RUL Regression
#   Predict cycles remaining before End of Life
#   Used for long-term replacement planning
#
# MODEL 3 — Condition Classification
#   Classify battery as Healthy / Degraded / Near EOL
#   Used for driver alerts and service notifications
#
# Training strategy:
#   Train on batteries B0005, B0006, B0007
#   Test on battery B0018 (completely unseen battery)
#   This simulates real BMW deployment — train on fleet, predict new battery
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13})

BASE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(BASE)
DATA    = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

EOL_SOH = 0.70

print("=" * 65)
print("  NOTEBOOK 3 — MODELLING & EVALUATION")
print("  EV Battery Health | BMW Group | NASA Dataset")
print("=" * 65)

# ── Load Engineered Data ──────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "battery_data_engineered.csv"))
print(f"\n  Loaded: {df.shape[0]} cycles from {df['battery_id'].nunique()} batteries")

# ── Feature and Target Columns ────────────────────────────────────────────────
ALL_FEATURES = [
    "cycle", "ambient_temp_c",
    "voltage_mean", "voltage_min", "voltage_std", "voltage_range",
    "current_mean", "current_std",
    "temp_mean", "temp_max", "temp_rise",
    "discharge_duration",
    "voltage_mean_roll_mean", "voltage_mean_roll_std",
    "temp_rise_roll_mean", "temp_rise_roll_std",
    "discharge_duration_roll_mean", "discharge_duration_roll_std",
    "capacity_ah_roll_mean", "capacity_ah_roll_std",
    "cap_fade_rate", "cycle_norm"
]

# ── Train / Test Split ────────────────────────────────────────────────────────
# IMPORTANT: We split by battery, NOT randomly
# Why? Random split would leak future cycles into training
# Real deployment: model trained on some batteries, predicts new ones

TRAIN_BATTERIES = ["B0005", "B0006", "B0007"]
TEST_BATTERY    = "B0018"

train = df[df["battery_id"].isin(TRAIN_BATTERIES)].dropna(subset=ALL_FEATURES)
test  = df[df["battery_id"] == TEST_BATTERY].dropna(subset=ALL_FEATURES)

X_train = train[ALL_FEATURES]
X_test  = test[ALL_FEATURES]

print(f"\n  Train: {len(X_train):,} cycles | Batteries: {TRAIN_BATTERIES}")
print(f"  Test : {len(X_test):,}  cycles | Battery : {TEST_BATTERY}")

# =============================================================================
# MODEL 1 — SoH REGRESSION
# =============================================================================
print("\n" + "=" * 65)
print("  MODEL 1 — SoH REGRESSION")
print("  Predict exact State of Health (0.7 to 1.0)")
print("=" * 65)

y_soh_train = train["SoH"]
y_soh_test  = test["SoH"]

# Random Forest Regressor
# n_estimators=300: 300 decision trees vote together
# max_depth=12: each tree can go 12 levels deep
# min_samples_leaf=3: each leaf needs at least 3 data points
#                     (prevents overfitting)
soh_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,  # for reproducibility
    n_jobs=-1         # use all CPU cores
)
soh_model.fit(X_train, y_soh_train)
soh_pred = soh_model.predict(X_test)

# Evaluation metrics
soh_rmse = np.sqrt(mean_squared_error(y_soh_test, soh_pred))
soh_mae  = mean_absolute_error(y_soh_test, soh_pred)
soh_r2   = r2_score(y_soh_test, soh_pred)

print(f"\n  RMSE : {soh_rmse:.4f}  ({soh_rmse*100:.2f}% average error)")
print(f"  MAE  : {soh_mae:.4f}  ({soh_mae*100:.2f}% average error)")
print(f"  R²   : {soh_r2:.4f}")
print(f"\n  Interpretation:")
print(f"  Average SoH prediction error = {soh_mae*100:.2f}%")
print(f"  If true SoH = 85%, model predicts ~{85 - soh_mae*100:.1f}% to {85 + soh_mae*100:.1f}%")

# =============================================================================
# MODEL 2 — RUL REGRESSION
# =============================================================================
print("\n" + "=" * 65)
print("  MODEL 2 — RUL REGRESSION")
print("  Predict cycles remaining before End of Life")
print("=" * 65)

y_rul_train = train["RUL"]
y_rul_test  = test["RUL"]

rul_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
rul_model.fit(X_train, y_rul_train)
rul_pred = rul_model.predict(X_test)

rul_rmse = np.sqrt(mean_squared_error(y_rul_test, rul_pred))
rul_mae  = mean_absolute_error(y_rul_test, rul_pred)
rul_r2   = r2_score(y_rul_test, rul_pred)

print(f"\n  RMSE : {rul_rmse:.2f} cycles")
print(f"  MAE  : {rul_mae:.2f} cycles")
print(f"  R²   : {rul_r2:.4f}")
print(f"\n  Interpretation:")
print(f"  On average predictions are {rul_mae:.1f} cycles off")
print(f"  If true RUL = 50 cycles, model predicts ~{50-rul_mae:.0f} to {50+rul_mae:.0f}")

# =============================================================================
# MODEL 3 — CONDITION CLASSIFICATION
# =============================================================================
print("\n" + "=" * 65)
print("  MODEL 3 — CONDITION CLASSIFICATION")
print("  Classify: Healthy / Degraded / Near EOL")
print("=" * 65)

# Encode string labels to numbers for sklearn
# LabelEncoder: Degraded=0, Healthy=1, Near EOL=2
le = LabelEncoder()
y_cond_train = le.fit_transform(train["condition"].astype(str))
y_cond_test  = le.transform(test["condition"].astype(str))

# class_weight="balanced" handles class imbalance
# (more "Near EOL" cycles than "Healthy" cycles in our data)
cond_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
cond_model.fit(X_train, y_cond_train)
cond_pred = cond_model.predict(X_test)

cond_acc = accuracy_score(y_cond_test, cond_pred)
print(f"\n  Accuracy : {cond_acc:.4f}")
print(f"\n{classification_report(y_cond_test, cond_pred, target_names=le.classes_)}")

# =============================================================================
# EVALUATION PLOTS
# =============================================================================
print("\n  Creating evaluation dashboard...")

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig)

# ── Plot 1: SoH Predicted vs Actual (scatter) ─────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_soh_test, soh_pred,
            alpha=0.6, color="#00B89F", s=25,
            edgecolors="white", linewidth=0.3)
mn = min(y_soh_test.min(), soh_pred.min())
mx = max(y_soh_test.max(), soh_pred.max())
ax1.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction")
ax1.set_xlabel("True SoH")
ax1.set_ylabel("Predicted SoH")
ax1.set_title(f"SoH: Predicted vs Actual\n"
              f"RMSE={soh_rmse:.4f} | R²={soh_r2:.3f}",
              fontweight="bold")
ax1.legend()

# ── Plot 2: SoH over cycles (predicted vs actual timeline) ────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(test["cycle"].values, y_soh_test.values,
         color="#1C1C1C", linewidth=2, label="True SoH")
ax2.plot(test["cycle"].values, soh_pred,
         color="#00B89F", linewidth=2,
         linestyle="--", label="Predicted SoH")
ax2.axhline(EOL_SOH, color="red", linestyle=":",
            lw=1.5, label="EOL threshold (70%)")
ax2.set_xlabel("Cycle")
ax2.set_ylabel("State of Health")
ax2.set_title("SoH Timeline — Battery B0018\n"
              "(Trained on B0005/06/07, tested on B0018)",
              fontweight="bold")
ax2.legend(fontsize=9)

# ── Plot 3: RUL Predicted vs Actual (scatter) ─────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(y_rul_test, rul_pred,
            alpha=0.6, color="#3498DB", s=25,
            edgecolors="white", linewidth=0.3)
mn = min(y_rul_test.min(), rul_pred.min())
mx = max(y_rul_test.max(), rul_pred.max())
ax3.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction")
ax3.set_xlabel("True RUL (cycles)")
ax3.set_ylabel("Predicted RUL (cycles)")
ax3.set_title(f"RUL: Predicted vs Actual\n"
              f"RMSE={rul_rmse:.1f} | R²={rul_r2:.3f}",
              fontweight="bold")
ax3.legend()

# ── Plot 4: Confusion Matrix ───────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
cm  = confusion_matrix(y_cond_test, cond_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4,
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            linewidths=1, annot_kws={"size": 12})
ax4.set_title(f"Condition Classification\n"
              f"Accuracy = {cond_acc:.3f}",
              fontweight="bold")
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")

# ── Plot 5: Feature Importances ────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
fi  = pd.Series(
    soh_model.feature_importances_,
    index=ALL_FEATURES
).nlargest(10).sort_values()
colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(fi)))
ax5.barh(fi.index, fi.values, color=colors_fi, edgecolor="white")
ax5.set_title("Top 10 Feature Importances\n(SoH Model)", fontweight="bold")
ax5.set_xlabel("Importance")

# ── Plot 6: RUL timeline ───────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(test["cycle"].values, y_rul_test.values,
         color="#1C1C1C", linewidth=2, label="True RUL")
ax6.plot(test["cycle"].values, rul_pred,
         color="#E74C3C", linewidth=2,
         linestyle="--", label="Predicted RUL")
ax6.axhline(30, color="orange", linestyle=":",
            lw=1.5, label="Alert zone (< 30 cycles)")
ax6.fill_between(test["cycle"].values,
                 0, 30, alpha=0.05, color="orange")
ax6.set_xlabel("Cycle")
ax6.set_ylabel("RUL (cycles)")
ax6.set_title("RUL Timeline — Battery B0018",
              fontweight="bold")
ax6.legend(fontsize=9)

plt.suptitle(
    "EV Battery Health Prediction — Model Evaluation Dashboard\n"
    "BMW Group Electric Vehicle Division | NASA Li-ion Dataset",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "model_01_evaluation_dashboard.png"),
            bbox_inches="tight")
plt.show()
print("  ✔ Saved: model_01_evaluation_dashboard.png")

# =============================================================================
# SAVE PREDICTIONS
# =============================================================================
out = test[["battery_id", "cycle", "capacity_ah", "SoH", "RUL",
            "condition"]].copy().reset_index(drop=True)
out["predicted_SoH"]       = soh_pred.round(4)
out["predicted_RUL"]       = rul_pred.round(0).astype(int)
out["predicted_condition"]  = le.inverse_transform(cond_pred)
out.to_csv(os.path.join(OUTPUTS, "battery_predictions.csv"), index=False)
print("  ✔ Saved: battery_predictions.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY — BMW GROUP DELIVERABLE")
print("=" * 65)
print(f"""
  MODEL 1 — State of Health Regression:
    RMSE : {soh_rmse:.4f} ({soh_rmse*100:.2f}% average error)
    MAE  : {soh_mae:.4f} ({soh_mae*100:.2f}% average error)
    R2   : {soh_r2:.4f}

  MODEL 2 — Remaining Useful Life Regression:
    RMSE : {rul_rmse:.2f} cycles
    MAE  : {rul_mae:.2f} cycles
    R2   : {rul_r2:.4f}

  MODEL 3 — Condition Classification:
    Accuracy : {cond_acc:.4f}
    Classes  : Healthy / Degraded / Near EOL

  BUSINESS IMPACT:
    SoH model predicts battery health with {soh_rmse*100:.2f}% RMSE error
    RUL model predicts remaining life within {rul_mae:.1f} cycles on average
    Condition model correctly classifies {cond_acc*100:.1f}% of battery states

  RECOMMENDATIONS FOR BMW GROUP:
    1. Deploy SoH model in BMW Battery Management System (BMS)
    2. Show real-time SoH on driver dashboard for accurate range
    3. Alert service team when SoH drops below 80%
    4. Schedule replacement when predicted RUL < 30 cycles
    5. Extend dataset to full BMW EV fleet for better generalisation
""")
print("  ✔ MODELLING COMPLETE")
