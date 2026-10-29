<div align="center">

# 🍔🛵 Food Delivery Time Prediction

**End-to-end ML project: messy real-world data → cleaned, engineered, tuned model → saved & reusable**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Cleaning-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Final%20Model-189AB4)
![Task](https://img.shields.io/badge/Task-Regression-success)
![R2](https://img.shields.io/badge/Test%20R%C2%B2-~0.83-brightgreen)

</div>

---

## ⚡ TL;DR

| | |
|---|---|
| **Goal** | Predict food delivery time (minutes) from information known at order time |
| **Data** | ~45.6k orders, 20 columns, real and messy |
| **Final model** | Tuned **XGBoost Regressor** |
| **Test R²** | **~0.83** (Adjusted R² ≈ 0.829) |
| **Baseline** | Linear Regression, test R² ≈ 0.57 |
| **Biggest data catch** | 431 rows with impossible distances (6,000–20,000 km) caused by lat/long sign errors |

## 📌 Contents
- [Problem](#-problem)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [Data Cleaning](#-data-cleaning)
- [EDA Findings](#-eda-key-findings)
- [Feature Engineering](#-feature-engineering)
- [Preprocessing](#-preprocessing)
- [Models & Results](#-models--results)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Business Recommendations](#-business-recommendations)
- [Using the Model](#-using-the-saved-model)
- [Tech Stack](#-tech-stack)
- [Future Work](#-future-work)

---

## 🎯 Problem

Apps like Zomato and Swiggy show an ETA before delivery. Accurate ETAs mean happier customers and better rider planning.

**ML formulation:** Supervised learning → **Regression**
- **Target:** `Time_taken(min)`
- **Inputs:** rider age & rating, distance, weather, traffic, vehicle, festival, city, order hour, etc.
- **Metrics:** R², Adjusted R², MSE, RMSE

## 📦 Dataset

~45,593 rows × 20 columns.

| Group | Columns |
|---|---|
| IDs | `ID`, `Delivery_person_ID` |
| Rider | `Delivery_person_Age`, `Delivery_person_Ratings` |
| Location | Restaurant & delivery latitude/longitude |
| Time | `Order_Date`, `Time_Ordered`, `Time_Order_picked` |
| Conditions | `Weatherconditions`, `Road_traffic_density` |
| Order / Vehicle | `Vehicle_condition`, `Type_of_order`, `Type_of_vehicle`, `multiple_deliveries` |
| Context | `Festival`, `City` |
| **Target** | `Time_taken(min)` |

## 🔄 Workflow

```
Business problem → Regression formulation
   → Data cleaning → EDA
   → Feature engineering (distance, order hour)
   → Encoding (ordinal + one-hot)
   → 80/20 split → StandardScaler (fit on train only)
   → Linear Reg → Decision Tree → Random Forest → XGBoost
   → RandomizedSearchCV (3-fold CV)
   → Evaluate → Save with joblib → Predict on new data
```

## 🧹 Data Cleaning

| Problem | Fix |
|---|---|
| `conditions ` prefix in `Weatherconditions` | Split and kept the weather word |
| `Time_taken(min)` stored as text (`(min) 24`) | Extracted number, cast to `int` |
| Missing values written as the string `"NaN"` (so `df.info()` showed no nulls) | Stripped whitespace, replaced with real `np.nan` |
| Extra spaces in text columns | `str.strip()` on all object columns |
| Numeric columns stored as `object` | `pd.to_numeric(errors="coerce")` |
| Ratings of 6 (max is 5) in 53 rows | Clipped to 5 |
| Missing numeric values | Median |
| Missing categorical values | Mode |
| Impossible GPS coordinates | Removed **431 rows** |

**Biggest catch:** after computing Haversine distance, some orders showed 6,000–20,000 km. Cause: stray negative signs in lat/long. After removal, all distances were under ~21 km. Final dataset: **~45,162 rows**.

## 🔍 EDA: Key Findings

**Target:** 10–54 min, median ≈ 26 min, roughly symmetric.

| Factor | Finding |
|---|---|
| 🚦 Traffic | Low ~21 → Medium ~26 → High ~27 → **Jam ~31 min** |
| 🌦️ Weather | Sunny fastest (~21 min), Cloudy ~28 min, fog/storm/wind/sandstorm slower |
| 🎉 Festival | No: ~25 min, **Yes: ~45 min** (almost double) |
| 🏙️ City type | **Semi-urban slowest (~49 min)** |
| 🕐 Time of day | Morning ~19 min, lunch ~26–27 min, **dinner (7–9 PM) ~31 min** |
| 👤 Rider age | 15–25 fastest; times rise above ~30 |
| ⭐ Rider rating | 4.5–5 rated riders fastest; only ~17% below 4.5 |
| 🛵 Vehicle / order type | Little effect |

Input features were not strongly correlated with each other, so multicollinearity was not a concern.

## 🛠️ Feature Engineering

| Feature | Method | Why |
|---|---|---|
| `distance_km` | Haversine on restaurant & delivery coordinates | Models struggle to derive distance from raw lat/long |
| `Order_hour` | `pd.to_datetime(...).dt.hour` | Captures lunch/dinner rush |

**Dropped:** `ID`, `Delivery_person_ID` (~13k unique values, no predictive meaning), `Order_Date`, raw coordinates, pickup time.

**Rejected experiment:** `Order_period` (Lunch/Dinner/Normal) from `Order_hour`. Adjusted R² moved only 0.8287 → 0.8288, so it was discarded. Tree models already find their own split points.

## ⚙️ Preprocessing

- **Ordinal:** `Road_traffic_density` → Low=0, Medium=1, High=2, Jam=3
- **One-Hot (`drop_first=True`):** weather, order type, vehicle type, festival, city
- **Split:** 80/20, `random_state=42` → ~36,129 train / ~9,033 test
- **Scaling:** `StandardScaler` fit on train only, then applied to test (avoids data leakage)

## 🤖 Models & Results

| Model | Train R² | Test R² | Verdict |
|---|---|---|---|
| Linear Regression (baseline) | ~0.58 | ~0.57 | Underfits, relationships are non-linear |
| Decision Tree (no depth limit) | 1.00 | ~0.67 | Overfits, memorised training data |
| Random Forest | ~0.97 | ~0.82 | Big jump, generalises well |
| Random Forest (tuned) | - | ~0.83 | After RandomizedSearchCV |
| XGBoost | - | ~0.82 | Boosting |
| **XGBoost (tuned)** ⭐ | ~0.88 | **~0.83** | **Final model**, Adj. R² ≈ 0.829 |

**Why XGBoost over Random Forest?** Tuned scores were nearly identical, so the choice also considered training time and compute cost. XGBoost also has a much smaller train/test gap than the Random Forest (0.88 vs 0.97 train), meaning less overfitting.

**Top features (Random Forest importance):** rider rating, rider age, distance, traffic density, multiple deliveries, festival.
**Least important:** order hour, vehicle condition, order type, vehicle type.

## 🎛️ Hyperparameter Tuning

- `RandomizedSearchCV`, 3-fold CV, 25 random combinations (75 fits) instead of a full grid (540 combinations × 3 = 1,620 fits)
- **Random Forest:** `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_features`
- **XGBoost:** `learning_rate`, `subsample`, `reg_alpha`, `reg_lambda`
- Best model refit on the full training set. **Test set was never used during tuning.**

## 💼 Business Recommendations

1. **Traffic jams:** reroute riders and show longer ETAs
2. **Festivals:** onboard extra delivery partners (time nearly doubles)
3. **Dinner rush (7–9 PM):** more riders plus rush-hour incentives
4. **Restaurants:** alert them before peak hours
5. **Bad weather:** add capacity and adjust ETAs
6. **Semi-urban areas:** investigate rider coverage
7. **Rider allocation:** use rider age/rating patterns for time-critical orders

## 💾 Using the Saved Model

```python
import joblib
import pandas as pd

model  = joblib.load("models/xgb_best.pkl")
scaler = joblib.load("models/standard_scaler.pkl")

# New order, encoded the same way as the training data
new_order = pd.DataFrame([...], columns=X_train.columns)

eta = model.predict(scaler.transform(new_order))
print(f"Estimated delivery time: {eta[0]:.1f} minutes")
```

Sanity check on unseen test rows: actual 19 → predicted ≈ 19; actual 40 → predicted ≈ 38.6.

## 🧰 Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost · joblib · Google Colab / Jupyter

## 🔮 Future Work

- Per-rider average delivery time feature (weighted by delivery count)
- Real road distance via a routing API instead of straight-line Haversine
- LightGBM / CatBoost and stacking
- Safe rider-ID features using target encoding with leakage control
- Deploy as FastAPI / Streamlit app for live ETA
- SHAP for explainability

## 📁 Repository Structure

```
food-delivery-time-prediction/
├── data/
├── notebooks/
│   └── Food_Delivery_Time_Prediction.ipynb
├── models/
│   ├── xgb_best.pkl
│   └── standard_scaler.pkl
├── README.md
└── requirements.txt
```
