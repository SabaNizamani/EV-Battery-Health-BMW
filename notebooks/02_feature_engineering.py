# =============================================================================
# EV Battery Health Prediction — Notebook 2: Feature Engineering
# Client: BMW Group — Electric Vehicle Division
# Dataset: NASA Ames Li-ion Battery Aging Dataset (PCoE)
# =============================================================================
# Goal: Transform raw battery measurements into ML-ready features
#       that capture degradation trends clearly and accurately.
#
# Framework:
#   1. Load processed data from EDA notebook
#   2. Add rolling statistics (smooth noisy signals)
#   3. Add capacity fade rate (how fast is it degrading?)
#   4. Normalise cycle position (where in lifetime is this battery?)
#   5. Create classification target (Healthy/Degraded/Near EOL)
#   6. Save final dataset ready for modelling
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13})

BASE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(BASE)
DATA    = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

print("=" * 65)
print("  NOTEBOOK 2 — FEATURE ENGINEERING")
print("  EV Battery Health | BMW Group | NASA Dataset")
print("=" * 65)

# ── Load Data ─────────────────────────────────────────────────────────────────
# Load the processed data saved by the EDA notebook
df = pd.read_csv(os.path.join(DATA, "battery_data_processed.csv"))
df = df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)

print(f"\n  Loaded: {df.shape[0]} cycles from {df['battery_id'].nunique()} batteries")

# Base features from raw sensor data
BASE_FEATURES = [
    "cycle", "ambient_temp_c",
    "voltage_mean", "voltage_min", "voltage_std", "voltage_range",
    "current_mean", "current_std",
    "temp_mean", "temp_max", "temp_rise",
    "discharge_duration"
]

# =============================================================================
# FEATURE 1 — Rolling Statistics (Window = 5 cycles)
# =============================================================================
# Problem: Raw sensor readings are noisy — they jump up and down randomly
# even though the underlying degradation is smooth.
#
# Solution: Rolling average smooths the noise by averaging
# the last 5 cycles instead of using just one cycle's reading.
#
# Why window of 5?
# - Too small (1-2): still noisy
# - Too large (20+): loses recent trends
# - 5 cycles: good balance for battery degradation pace

WINDOW = 5

print(f"\n  Adding rolling features (window = {WINDOW} cycles)...")

ROLL_TARGETS = [
    "voltage_mean",
    "temp_rise",
    "discharge_duration",
    "capacity_ah"
]

for feat in ROLL_TARGETS:
    grp = df.groupby("battery_id")[feat]

    # Rolling MEAN — smoothed average of last 5 cycles
    df[f"{feat}_roll_mean"] = grp.transform(
        lambda x: x.rolling(WINDOW, min_periods=1).mean()
    )

    # Rolling STD — how much the reading varied in last 5 cycles
    # High std = unstable = degrading faster
    df[f"{feat}_roll_std"] = grp.transform(
        lambda x: x.rolling(WINDOW, min_periods=1).std().fillna(0)
    )

print(f"  ✔ Added {len(ROLL_TARGETS) * 2} rolling features")

# =============================================================================
# FEATURE 2 — Capacity Fade Rate
# =============================================================================
# How fast is the battery losing capacity?
# Fade rate = capacity this cycle - capacity last cycle
# Negative value = capacity is dropping (degrading)
# Steeper negative = degrading faster = more urgent

print("\n  Adding capacity fade rate...")

df["cap_fade_rate"] = df.groupby("battery_id")["capacity_ah"].transform(
    lambda x: x.diff().fillna(0)
)

print(f"  ✔ Capacity fade rate added")
print(f"  Mean fade rate: {df['cap_fade_rate'].mean():.6f} Ah per cycle")
print(f"  (Negative = losing capacity)")

# =============================================================================
# FEATURE 3 — Cycle Normalised
# =============================================================================
# Different batteries last different numbers of cycles before EOL.
# Battery B0005 might last 125 cycles, B0007 might last 168 cycles.
#
# To compare them fairly, we normalise cycle position:
# 0.0 = very first cycle (brand new)
# 1.0 = very last recorded cycle (near end of life)
#
# This tells the model WHERE the battery is in its lifetime.

print("\n  Adding normalised cycle position...")

df["cycle_norm"] = df.groupby("battery_id")["cycle"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
)

print(f"  ✔ Cycle normalised (0 = new, 1 = end of life)")

# =============================================================================
# FINAL FEATURE LIST
# =============================================================================
ALL_FEATURES = BASE_FEATURES + [
    "voltage_mean_roll_mean",
    "voltage_mean_roll_std",
    "temp_rise_roll_mean",
    "temp_rise_roll_std",
    "discharge_duration_roll_mean",
    "discharge_duration_roll_std",
    "capacity_ah_roll_mean",
    "capacity_ah_roll_std",
    "cap_fade_rate",
    "cycle_norm"
]

print(f"\n  Base features     : {len(BASE_FEATURES)}")
print(f"  Rolling features  : {len(ROLL_TARGETS) * 2}")
print(f"  Computed features : 2 (fade_rate + cycle_norm)")
print(f"  Total features    : {len(ALL_FEATURES)}")

# =============================================================================
# VISUALISE — Rolling Feature Effect
# =============================================================================
print("\n  Plotting rolling feature effect (raw vs smoothed)...")

b5 = df[df["battery_id"] == "B0005"].copy()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Raw voltage vs smoothed voltage
axes[0].plot(b5["cycle"], b5["voltage_mean"],
             alpha=0.4, color="#E74C3C", linewidth=1, label="Raw voltage mean")
axes[0].plot(b5["cycle"], b5["voltage_mean_roll_mean"],
             color="#00B89F", linewidth=2.5, label="Rolling mean (5 cycles)")
axes[0].set_title("Raw vs Smoothed Voltage Signal\nBattery B0005",
                  fontweight="bold")
axes[0].set_xlabel("Cycle")
axes[0].set_ylabel("Mean Voltage (V)")
axes[0].legend()

# Capacity fade rate
axes[1].bar(b5["cycle"], b5["cap_fade_rate"],
            color=["#E74C3C" if v < -0.002 else "#00B89F"
                   for v in b5["cap_fade_rate"]],
            alpha=0.7, width=1)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Capacity Fade Rate Per Cycle\n"
                  "(Red = fast degradation, Green = slow/stable)",
                  fontweight="bold")
axes[1].set_xlabel("Cycle")
axes[1].set_ylabel("Capacity Change (Ah)")

plt.suptitle("Feature Engineering — Rolling Smoothing & Fade Rate",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "fe_01_rolling_features.png"),
            bbox_inches="tight")
plt.show()
print("  ✔ Saved: fe_01_rolling_features.png")

# =============================================================================
# FEATURE IMPORTANCE PREVIEW — Correlation with SoH
# =============================================================================
corr_with_soh = (df[ALL_FEATURES + ["SoH"]]
                 .corr()["SoH"]
                 .drop("SoH")
                 .abs()
                 .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(10, 8))
colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(corr_with_soh)))
ax.barh(corr_with_soh.index, corr_with_soh.values,
        color=colors_bar, edgecolor="white")
ax.set_title("Feature Correlation with SoH (Absolute)\n"
             "Higher = More Important for Prediction",
             fontweight="bold")
ax.set_xlabel("Absolute Pearson Correlation with SoH")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "fe_02_feature_importance_preview.png"),
            bbox_inches="tight")
plt.show()
print("  ✔ Saved: fe_02_feature_importance_preview.png")

# =============================================================================
# SAVE ENGINEERED DATASET
# =============================================================================
KEEP_COLS = (["battery_id", "cycle"] + ALL_FEATURES +
             ["capacity_ah", "SoH", "RUL", "condition"])
df_final  = df[KEEP_COLS].copy()

df_final.to_csv(os.path.join(DATA, "battery_data_engineered.csv"), index=False)

print(f"\n  ✔ Saved: battery_data_engineered.csv")
print(f"  Final shape: {df_final.shape}")

print("\n" + "=" * 65)
print("  FEATURE ENGINEERING COMPLETE — SUMMARY")
print("=" * 65)
print(f"""
  Features created:

  Rolling mean (x4 sensors)  : Smoothed signal — removes noise
  Rolling std  (x4 sensors)  : Instability measure
  Cap fade rate               : How fast is battery losing capacity
  Cycle normalised            : Where in lifetime is this battery

  Key insight:
  Rolling features are MUCH better predictors than raw readings
  because they capture the TREND not just a single noisy point.
  This is the same approach used in real BMW BMS systems.

  Next step: Modelling (03_modelling.py)
""")
