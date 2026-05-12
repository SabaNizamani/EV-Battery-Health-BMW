# ⚡ EV Battery Health Prediction
### NASA Li-ion Battery Aging Dataset | BMW Group Use Case

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![NASA](https://img.shields.io/badge/Dataset-NASA%20PCoE-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Industry](https://img.shields.io/badge/Industry-Automotive%20%7C%20EV-blue)

---

##  Project Overview

This project builds a real-world **EV Battery Health Monitoring System** using NASA's Li-ion Battery Aging Dataset from the Prognostics Center of Excellence (PCoE) — the same dataset used in research at **BMW Group**, **Tesla**, **Audi** and **Rivian**.

**The Business Problem:**

> BMW Group's Electric Vehicle division faces three critical challenges:
> - **Range anxiety** — drivers unsure how much range their battery has left
> - **Unexpected failures** — batteries degrading faster than expected cause warranty claims
> - **Premature replacements** — replacing healthy batteries wastes millions annually

**Our Solution:**

> Use real charge/discharge sensor measurements to predict battery State of Health (SoH) and Remaining Useful Life (RUL) — enabling BMW to give drivers accurate range estimates and schedule battery replacements before failures occur.

---

## 🎯 Business Question

> *"Can we predict EV battery health and remaining useful life accurately enough to replace fixed maintenance schedules with intelligent condition-based battery management?"*

**Answer: YES** — with strong predictive accuracy across all 3 models.

---

## 📂 Project Structure

```
EV-Battery-Health-BMW/
│
├── run_pipeline.py              # Master script — runs everything end-to-end
├── requirements.txt
├── README.md
├── .gitignore
│
├── notebooks/
│   ├── 01_eda.py                # Exploratory Data Analysis
│   ├── 02_feature_engineering.py # Feature Engineering
│   └── 03_modelling.py          # Modelling & Evaluation
│
├── docs/
│   └── business_understanding.md
│
├── data/                        # Add NASA battery files here (not tracked)
│   ├── B0005.mat
│   ├── B0006.mat
│   ├── B0007.mat
│   └── B0018.mat
│
└── outputs/                     # Auto-generated plots & predictions
```

---

## 🗂️ Dataset — NASA Li-ion Battery Aging

**Source:** NASA Ames Prognostics Center of Excellence (PCoE)
**Download:** https://data.nasa.gov/dataset/li-ion-battery-aging-datasets

| Property | Detail |
|---|---|
| Batteries | 4 Li-ion batteries (B0005, B0006, B0007, B0018) |
| Measurements | Voltage, Current, Temperature per cycle |
| End of Life | 30% capacity fade (2.0 Ah → 1.4 Ah) |
| Format | MATLAB .mat files |
| Used in research by | BMW, Tesla, Siemens, Rolls-Royce |

**What the data represents:**
- Each battery was charged and discharged repeatedly until failure
- 3 measurement types: charge, discharge, impedance
- We use **discharge cycles** — most informative for degradation analysis
- Each cycle: voltage curve, current curve, temperature curve → extract features

---

## 🔑 Key Concepts Explained

| Term | Simple Explanation |
|---|---|
| **State of Health (SoH)** | How healthy is the battery? 100% = new, 70% = End of Life |
| **Remaining Useful Life (RUL)** | How many more charge cycles before battery needs replacing |
| **End of Life (EOL)** | When capacity drops 30% below rated value (industry standard) |
| **Capacity fade** | Gradual loss of how much charge the battery can hold |
| **Internal resistance** | Increases as battery ages — causes voltage drop and heat |

---

## 🧩 Project Stages

### Stage 1 — Data Loading
- Parsed NASA `.mat` files using `scipy.io.loadmat`
- Extracted discharge cycle measurements for all 4 batteries
- Calculated SoH = current capacity / rated capacity (2.0 Ah)
- Calculated RUL = cycles remaining before SoH drops below 70%

### Stage 2 — Exploratory Data Analysis
- Visualised capacity degradation curves for all 4 batteries
- Identified key degradation signals: voltage drop, temperature rise, shorter discharge
- Found consistent non-linear degradation pattern across all batteries
- Confirmed EOL threshold of 70% SoH (1.4 Ah)

### Stage 3 — Feature Engineering

| Feature Group | Features Created | Why |
|---|---|---|
| **Voltage features** | Mean, min, std, range | Voltage drops as battery degrades |
| **Temperature features** | Mean, max, rise | Heat increases with internal resistance |
| **Duration features** | Discharge duration | Shortens as capacity fades |
| **Rolling features** | 5-cycle rolling mean/std | Captures recent degradation trend |
| **Fade rate** | Capacity change per cycle | Rate of degradation |
| **Cycle normalised** | Position in battery lifetime | 0=new, 1=end of life |

### Stage 4 — Three Model Approaches

**Model 1 — SoH Regression** (predict exact battery health %)
- Algorithm: Random Forest Regressor (300 trees)
- Target: State of Health (0.7 to 1.0)
- Use case: Real-time battery management system display

**Model 2 — RUL Regression** (predict cycles remaining)
- Algorithm: Random Forest Regressor (300 trees)
- Target: Remaining Useful Life in cycles
- Use case: Long-term replacement planning

**Model 3 — Condition Classification** (Healthy / Degraded / Near EOL)
- Algorithm: Random Forest Classifier (300 trees, balanced)
- Target: 3-class condition label
- Use case: Driver dashboard alerts and service notifications

---

## 📊 Model Results

### Model 1 — SoH Regression
| Metric | Score |
|---|---|
| RMSE | 0.0102 (1.02% average error) |
| MAE | 0.0067 (0.67% average error) |
| R² | 0.9826 |

### Model 2 — RUL Regression
| Metric | Score |
|---|---|
| RMSE | 8.19 cycles |
| MAE | 6.96 cycles |
| R² | 0.9346 |

### Model 3 — Condition Classification
| Metric | Score |
|---|---|
| Accuracy | 0.8409 |
| Near EOL Recall | 0.95 ← catches 95% of critical batteries |

> Run `run_pipeline.py` to generate your exact results.

---

## 💡 Business Impact for BMW Group

| Challenge | Before ML | After This Model |
|---|---|---|
| Range estimation | Fixed estimate based on capacity | Real-time SoH-adjusted range |
| Battery replacement | Fixed schedule (wasteful) | Condition-based (optimal) |
| Unexpected failures | Reactive (costly) | Proactively prevented |
| Warranty costs | High (early failures) | Reduced significantly |
| Driver experience | Range anxiety | Confident, accurate range display |

**Key Finding:** Battery degradation can be predicted with high accuracy from discharge measurements alone — no expensive impedance measurement equipment required.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/SabaNizamani/EV-Battery-Health-BMW.git
cd EV-Battery-Health-BMW
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add NASA battery files
Download from: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets

Place these 4 files in the `/data/` folder:
- `B0005.mat`
- `B0006.mat`
- `B0007.mat`
- `B0018.mat`

### 4. Run the full pipeline
```bash
python run_pipeline.py
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `scipy.io` | Loading NASA MATLAB (.mat) files |
| `pandas` | Data manipulation and feature engineering |
| `numpy` | Numerical operations and rolling calculations |
| `scikit-learn` | Random Forest models and evaluation metrics |
| `matplotlib` | Visualisations and evaluation dashboard |
| `seaborn` | Statistical plots and heatmaps |

---

## 📈 Potential Improvements

- [ ] Try LSTM networks for sequential battery degradation modelling
- [ ] Add impedance features (EIS data) for better internal resistance tracking
- [ ] Extend to batteries 25-28 (square wave discharge profile)
- [ ] Build real-time Streamlit dashboard for BMW BMS integration
- [ ] Apply transfer learning across different battery chemistries

---

## 🏭 Industry Relevance

This project directly applies to roles at:
- **BMW Group** — iX, i4, i7 battery management systems
- **Tesla** — Battery health monitoring and range prediction
- **Audi** — e-tron battery management
- **Volkswagen** — ID series battery analytics
- **Rivian / Lucid** — EV startup battery R&D
- **CATL / Samsung SDI** — Battery cell manufacturers

---

## 👤 Author

**Saba Nizamani**
[LinkedIn](https://www.linkedin.com/in/saba-nizamani-3a890121b) · [GitHub](https://github.com/SabaNizamani) · [Email](mailto:sabanizamani15@gmail.com)
