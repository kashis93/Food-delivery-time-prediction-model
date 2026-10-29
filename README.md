<div align="center">

# 🍔🛵 Food Delivery Time Prediction

### An end-to-end Machine Learning project: from business problem to saved model

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Cleaning-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-189AB4)
![Task](https://img.shields.io/badge/Task-Regression-success)
![R2](https://img.shields.io/badge/Test%20R%C2%B2-~0.83-brightgreen)

*"Your order will arrive in 35 minutes."* — how does Zomato / Swiggy know that?
This project builds a model that predicts it. 🚀

</div>

---

## 📌 Table of Contents
1. [Business Problem](#-business-problem)
2. [ML Problem Formulation](#-ml-problem-formulation)
3. [Dataset](#-dataset)
4. [Project Workflow](#-project-workflow)
5. [Data Cleaning](#-data-cleaning)
6. [Exploratory Data Analysis & Key Findings](#-exploratory-data-analysis--key-findings)
7. [Feature Engineering](#-feature-engineering)
8. [Preprocessing](#-preprocessing)
9. [Modeling & Results](#-modeling--results)
10. [Hyperparameter Tuning](#-hyperparameter-tuning)
11. [Business Recommendations](#-business-recommendations)
12. [Saving & Using the Model](#-saving--using-the-model)
13. [Concepts Learned / Interview Notes](#-concepts-learned--interview-notes)
14. [Tech Stack](#-tech-stack)
15. [Future Work](#-future-work)

---

## 🎯 Business Problem

Food delivery apps show an **estimated delivery time** (25, 35, 45 mins...) *before* the order is delivered.
Accurate estimates mean happier customers, better rider planning and fewer complaints.

> **Can we predict how long a food order will take to be delivered, using information available when the order is placed?**

## 🧠 ML Problem Formulation

| Item | Description |
|---|---|
| **Learning type** | Supervised Machine Learning |
| **Problem type** | **Regression** (target is a continuous value in minutes) |
| **Target (output) feature** | `Time_taken(min)` |
| **Input features** | Rider age & rating, distance, weather, traffic, vehicle, festival, city, order hour, etc. |
| **Main metrics** | R², Adjusted R², MSE, RMSE |

---

## 📦 Dataset

A **real-life, messy dataset** (not a clean textbook one), which made the project far more realistic.

- **Rows:** ~45,593 orders
- **Columns:** 20

| Group | Columns |
|---|---|
| IDs | `ID`, `Delivery_person_ID` |
| Rider | `Delivery_person_Age`, `Delivery_person_Ratings` |
| Location | Restaurant latitude/longitude, Delivery latitude/longitude |
| Time | `Order_Date`, `Time_Ordered`, `Time_Order_picked` |
| Conditions | `Weatherconditions`, `Road_traffic_density` |
| Order/Vehicle | `Vehicle_condition`, `Type_of_order`, `Type_of_vehicle`, `multiple_deliveries` |
| Context | `Festival`, `City` |
| **Target** | `Time_taken(min)` |

---

## 🔄 Project Workflow

```
Business Problem
      ↓
ML Problem (Supervised → Regression)
      ↓
Data Collection & Understanding
      ↓
Data Cleaning  →  EDA (business + ML insights)
      ↓
Feature Engineering (Distance, Order Hour)
      ↓
Encoding (Ordinal + One-Hot)
      ↓
Train / Test Split (80/20)  →  Feature Scaling
      ↓
Model Training: Linear Regression → Decision Tree → Random Forest → XGBoost
      ↓
Hyperparameter Tuning (RandomizedSearchCV + Cross-Validation)
      ↓
Evaluation  →  Save Model (joblib)  →  Predict on new data
```

---

## 🧹 Data Cleaning

Real data was dirty. Problems found and fixed:

| Problem found | Fix applied |
|---|---|
| `Weatherconditions` had a `conditions ` prefix (e.g. `conditions Sunny`) | Split on space and kept the weather word |
| `Time_taken(min)` stored as text like `(min) 24` | Extracted the number and cast to `int` |
| Missing values were written as the **string** `"NaN"`, so `df.info()` falsely showed "no nulls" | Stripped whitespace, then replaced with real `np.nan` |
| Leading/trailing spaces in text columns | `str.strip()` on all object columns |
| Numeric columns stored as `object` (age, ratings, multiple deliveries) | `pd.to_numeric(..., errors="coerce")` |
| **Ratings of 6** (max is 5) in 53 rows, clearly entry errors | Clipped to a max of 5 |
| Missing numeric values | Filled with **median** (robust to outliers) |
| Missing categorical values | Filled with **mode** |
| Invalid GPS coordinates (see below) | Removed bad rows |

### 🚨 Biggest data-quality catch: impossible distances
After computing the distance with the Haversine formula, some orders showed **~6,000 to 20,000 km** deliveries, physically impossible.
- Root cause: **sign errors in latitude/longitude** during data collection (a stray negative sign teleports a point across the globe).
- **431 rows** were removed. Valid distances were all under ~21 km.
- Final dataset: **~45,162 rows**.

---

## 🔍 Exploratory Data Analysis & Key Findings

EDA served two purposes: **business insight** (better decisions) and **ML insight** (better features).

### 🎯 Target: delivery time
- Range: about **10 to 54 minutes**, median around **26 min**
- Mean ≈ median, so the distribution is fairly symmetric, with no extreme outliers

### 🚦 Road traffic density (strongest effect)
| Traffic | Avg. delivery time |
|---|---|
| Low | ~21 min |
| Medium | ~26 min |
| High | ~27 min |
| Jam | **~31 min** |

### 🌦️ Weather
- **Sunny:** ~21 min (fastest)
- **Cloudy:** ~28 min
- Worse weather (fog, stormy, windy, sandstorms) is noticeably slower

### 🎉 Festival
| Festival day? | Avg. time |
|---|---|
| No | ~25 min |
| **Yes** | **~45 min** (almost double!) |

### 🏙️ City type
- **Semi-urban** areas averaged ~**49 min**, the slowest
- Likely fewer delivery partners or longer distances

### 🕐 Time of day
- Morning (~8–9 AM): ~19 min (fastest)
- Lunch (~11 AM–2 PM): ~26–27 min
- **Dinner (7–9 PM): ~31 min** (rush hour)

### 👤 Rider characteristics
- **Age:** riders around 15–25 delivered faster; times rise noticeably for riders above ~30
- **Ratings:** riders rated **4.5–5** are the fastest. Only ~17% of riders are below 4.5 and ~2% below 4, so ratings vary little overall
- **Vehicle type & order type:** little difference in delivery time

### 🔗 Correlation
- Input features were **not highly correlated with each other** (good for linear models, since no multicollinearity)
- Age (positive) and ratings (negative) show moderate correlation with the target

> 💡 **Insight:** a business insight survives even if the ML model rejects a feature. EDA alone created business value without any model.

---

## 🛠️ Feature Engineering

| New feature | How | Why |
|---|---|---|
| **`distance_km`** | **Haversine formula** on restaurant and delivery coordinates | Raw lat/long are hard for models to turn into distance. Distance directly affects time |
| **`Order_hour`** | Hour extracted from `Time_ordered` using `pd.to_datetime(...).dt.hour` | Captures lunch/dinner rush patterns |

**Dropped:** `ID`, `Delivery_person_ID` (unique identifiers with no predictive meaning; ~13k unique rider IDs would explode on encoding), `Order_Date`, the raw coordinates (replaced by distance) and the pickup time.

### 🧪 Experiment: `Order_period` feature (Lunch / Dinner / Normal)
- Created from `Order_hour` and tested with **Adjusted R²**
- **Result: no meaningful improvement** (Adjusted R² ≈ 0.8287 → ≈ 0.8288), so the feature was **rejected**
- Reason: tree-based models already find the best split thresholds on their own. Hand-made time buckets help **linear** models, not trees

> ✅ Rule followed: *only keep a new feature if it measurably improves the model.*

---

## ⚙️ Preprocessing

### 🔢 Encoding
| Type | Features | Method |
|---|---|---|
| **Ordinal** (has natural order) | `Road_traffic_density` | Low = 0, Medium = 1, High = 2, Jam = 3 |
| **Nominal** (no order) | Weather, Type of order, Type of vehicle, Festival, City | **One-Hot Encoding** with `drop_first=True` |

### ✂️ Train/Test split first, then scale
- **80% train / 20% test** (`random_state=42`)
- Rows: ~36,129 train and ~9,033 test
- `StandardScaler` was **fit on training data only**; the test set was only **transformed**

> ⚠️ **Why split before scaling?** Scaling before the split leaks test-set mean/std into training. That is **data leakage**, which makes the test score misleadingly optimistic.

The fitted scaler was saved with `joblib` so new data can be scaled identically.

---

## 🤖 Modeling & Results

| Model | Train R² | Test R² | Notes |
|---|---|---|---|
| **Linear Regression** (baseline) | ~0.58 | ~0.57 | **Underfitting.** Relationships are non-linear |
| **Decision Tree** (no depth limit) | **1.00** | ~0.67 | **Overfitting.** Memorised the training data |
| **Random Forest** | ~0.97 | ~0.82 | Big jump, good generalisation |
| **Random Forest (tuned)** | n/a | **~0.83** | After RandomizedSearchCV |
| **XGBoost** | n/a | ~0.82 | Sequential boosting |
| **XGBoost (tuned)** | ~0.88 | **~0.83** | Final model. Adjusted R² ≈ **0.829** |

### 📖 The story the numbers tell
1. **Linear Regression → ~57%**: a straight line cannot capture the non-linear relationships (e.g. low distance but heavy traffic is still slow).
2. **Decision Tree → Train 1.0, Test ~0.67**: a classic **overfit**. Perfect on training data, much weaker on unseen data.
3. **Random Forest / XGBoost → ~83%**: ensembles fix overfitting (bagging reduces variance, boosting corrects earlier errors).
4. Tuned Random Forest and tuned XGBoost performed almost the same, so the final pick also weighs **training time and compute cost**. Tuned XGBoost was saved as the final model.

### 🌟 Feature importance (Random Forest, approximate)
Most important drivers: **Delivery person rating**, **Delivery person age**, **Distance**, **Road traffic density**, **Multiple deliveries**, **Festival**.
Least important: Order hour, Vehicle condition, Type of order/vehicle. Dropping these would barely change performance.

---

## 🎛️ Hyperparameter Tuning

- **Parameters** are learned from data (e.g. regression weights). **Hyperparameters** are set by us before training (e.g. `n_estimators`, `max_depth`, `learning_rate`)
- Used **RandomizedSearchCV** with **3-fold cross-validation** (25 random combinations → 75 fits) instead of an exhaustive Grid Search (540 combinations × 3 folds = 1,620 fits)
- Searched: `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_features` (RF); `learning_rate`, `subsample`, `reg_alpha`, `reg_lambda` (XGBoost)
- After picking the best combination, the model is **refit on the full training set**
- The **test set was never touched** during tuning, only for the final report

---

## 💼 Business Recommendations

From EDA findings, a delivery manager could:

1. 🚦 **Traffic jams:** assign alternate routes and show longer ETAs when traffic is heavy
2. 🎉 **Festivals:** onboard more delivery partners (delivery time almost doubles)
3. 🌆 **Dinner rush (7–9 PM):** schedule more riders and offer rush-hour incentives
4. 🍳 **Restaurants:** warn restaurants in advance to prepare for peak hours
5. 🌧️ **Bad weather:** add rider capacity and adjust ETAs
6. 🏘️ **Semi-urban areas:** investigate delivery partner coverage
7. 🧑 **Rider allocation:** consider rider age/rating patterns when assigning long or time-critical orders

---

## 💾 Saving & Using the Model

```python
import joblib
import pandas as pd

# Load saved artifacts
model  = joblib.load("xgb_best.pkl")
scaler = joblib.load("standard_scaler.pkl")

# New order (already encoded the same way as the training data)
new_order = pd.DataFrame([...], columns=X_train.columns)

# Scale with the TRAINING mean/std, then predict
new_scaled = scaler.transform(new_order)
eta = model.predict(new_scaled)
print(f"Estimated delivery time: {eta[0]:.1f} minutes")
```

Sample check on unseen test rows: predictions were close to actual values (e.g. actual 19 vs predicted ≈ 19, and actual 40 vs predicted ≈ 38.6).

---

## 🎓 Concepts Learned / Interview Notes

<details>
<summary><b>Why not feed raw dirty data to the model?</b></summary>
Garbage in, garbage out. The model treats data as the "ultimate truth". Missing values break math operations, wrong values teach wrong patterns, duplicates bias the loss function, and redundant features add noise.
</details>

<details>
<summary><b>Split first or scale first? Why?</b></summary>
<b>Split first, then scale.</b> Scaling the whole dataset leaks information about the test set (data leakage). Fit the scaler on train only, then transform test.
</details>

<details>
<summary><b>Ordinal vs Nominal encoding?</b></summary>
Ordinal when categories have a natural order (Low &lt; Medium &lt; High &lt; Jam). One-Hot when they do not (weather, city). Using 1, 2, 3 for unordered categories would feed the model a false ordering.
</details>

<details>
<summary><b>Why drop_first=True in one-hot encoding?</b></summary>
Avoids perfect linear dependency (the dummy variable trap), which hurts linear models and adds redundant features otherwise.
</details>

<details>
<summary><b>Do tree models need feature scaling?</b></summary>
No. Trees split on thresholds, not distances or gradients. Linear/Logistic Regression, KNN, SVM, PCA and Neural Nets do need scaling.
</details>

<details>
<summary><b>R² vs Adjusted R²?</b></summary>
R² never decreases when you add features, even useless ones. Adjusted R² penalises extra features, so it only rises when a new feature genuinely helps. Use it when testing feature engineering.
</details>

<details>
<summary><b>Why engineer Distance when lat/long already exist?</b></summary>
The Haversine relationship is a complex trigonometric formula that even non-linear models struggle to discover from raw coordinates. Handing the model the distance directly is far more effective.
</details>

<details>
<summary><b>Why did Decision Tree show R² = 1 on train?</b></summary>
With no depth limit it memorises the training data including noise, a textbook case of overfitting. Ensembles (Random Forest, XGBoost) reduce this.
</details>

<details>
<summary><b>Bagging vs Boosting?</b></summary>
Bagging (Random Forest) trains trees independently in parallel and averages them to reduce variance. Boosting (XGBoost) trains trees sequentially, each correcting the previous ensemble's errors.
</details>

<details>
<summary><b>Why does a business insight survive even if ML rejects a feature?</b></summary>
EDA findings (e.g. festivals double delivery time) directly guide business decisions regardless of whether the feature improves the model.
</details>

---

## 🧰 Tech Stack

- **Language:** Python
- **Data:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **ML:** scikit-learn (Linear Regression, Decision Tree, Random Forest, RandomizedSearchCV, StandardScaler, metrics), XGBoost
- **Persistence:** joblib (`.pkl`)
- **Environment:** Google Colab / Jupyter Notebook

---

## 🔮 Future Work

- 🏍️ **Rider smoothness feature:** per-rider average delivery time with delivery count as a reliability weight
- 🌐 Replace straight-line distance with **real road distance / routing API**
- 📉 Try LightGBM / CatBoost and ensemble stacking
- 🧪 Add rider ID based features safely (target encoding with leakage control)
- 🚀 Deploy as an API (FastAPI / Flask) or a Streamlit app for live ETA prediction
- 📊 Add SHAP for deeper model explainability

---

## 📁 Suggested Repository Structure

```
food-delivery-time-prediction/
├── data/                      # dataset (CSV)
├── notebooks/
│   └── Food_Delivery_Time_Prediction.ipynb
├── models/
│   ├── xgb_best.pkl
│   └── standard_scaler.pkl
├── README.md
└── requirements.txt
```

---

<div align="center">

### ⭐ If you found this useful, give the repo a star!

Made with ❤️ while learning end-to-end Machine Learning

</div>
