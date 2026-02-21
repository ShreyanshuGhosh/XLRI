# 🎬 StreamMax OTT — Engagement Fatigue Prediction

> **StrategiX 2.0 Hackathon** · Marvix Analytics · Data Science Competition

Predicting which StreamMax subscribers are at risk of disengagement — **30 days before they churn** — using behavioral data and an XGBoost-powered stacking ensemble.

---

## 📌 Problem Statement

StreamMax, a fast-growing OTT video streaming platform, loses subscribers before it can identify them as at-risk. Traditional churn models flag users only after complete disengagement — by which point re-engagement costs 5–10× more than proactive retention.

**Our goal:** Build a model that detects *engagement fatigue* early, outputting a probability score (0–1) for each user indicating their risk of disengagement within the next 30 days.

**Evaluation metric:** AUC-ROC

---

## 📁 Repository Structure

```
├── TeamName_XGBoost_Pipeline_Final.ipynb   # Main analysis & model notebook
├── TeamName_Predictions.csv                # Final predictions for 2,000 test users
├── TeamName_Presentation.pptx              # Competition presentation (6 slides)
├── data/
│   ├── ott_train.csv                       # Training data — 8,000 users (with labels)
│   └── ott_test.csv                        # Test data   — 2,000 users (no labels)
└── README.md
```

---

## 📊 Dataset

| File | Rows | Target |
|---|---|---|
| `ott_train.csv` | 8,000 | `fatigue_label` (0 = Engaged, 1 = At Risk) |
| `ott_test.csv` | 2,000 | — (to be predicted) |

### Key Features

| Feature | Description |
|---|---|
| `tenure_days` | Days since user first subscribed (7–1,095) |
| `subscription_tier` | Basic / Standard / Premium |
| `avg_daily_minutes_last_7d` | Avg watch time per day — last week |
| `avg_daily_minutes_last_30d` | Avg watch time per day — last month |
| `sessions_last_7d / _30d` | Number of viewing sessions |
| `days_since_last_session` | Recency signal — days since last login |
| `avg_completion_rate` | % of content finished (0–1) |
| `binge_sessions_last_30d` | Sessions with 3+ episodes or 2+ hours |
| `recommendation_click_rate` | % of recommended content clicked |
| `unique_genres_watched_30d` | Content diversity (1–15 genres) |
| `peak_hour_viewing_pct` | % of viewing during 7–11 PM |
| `original_content_pct` | % watch time on StreamMax originals |

---

## 🧠 Methodology

### 1. Exploratory Data Analysis
- Target distribution & class balance
- Feature distributions: Engaged vs At Risk
- Boxplots, correlation heatmap, outlier detection (IQR)
- Viewing decline pattern (7d vs 30d trend)
- Tenure cohort analysis & segmentation heatmap by subscription tier

### 2. Feature Engineering (40+ features)
Raw features extended with domain-informed signals:

| Category | Examples |
|---|---|
| **Viewing trend** | `viewing_trend_abs`, `viewing_trend_ratio` (7d vs 30d) |
| **Recency decay** | `recency_decay_3d/7d/14d` — exponential penalty for absence |
| **Session intensity** | `mins_per_session_7d`, `sessions_per_week` |
| **Binge behaviour** | `binge_ratio`, `zero_binge` flag |
| **Interaction features** | `absent_x_low_completion`, `sessions_x_completion` |
| **Risk flag count** | Aggregated count of red-flag signals per user |
| **Log/sqrt transforms** | Reduces skewness in session/minute counts |
| **Tenure buckets** | Lifecycle stage (new → 3-year subscriber) |

### 3. Feature Selection
XGBoost importance scores used to drop noisy features below threshold `0.003`, retaining the most predictive subset.

### 4. Hyperparameter Tuning
Two-stage `RandomizedSearchCV` with 5-fold stratified CV:
- **Stage 1:** Broad search — 100 random combinations across 11 parameters
- **Stage 2:** Early stopping — finds optimal `n_estimators` automatically

Key parameters tuned: `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `colsample_bylevel`, `colsample_bynode`, `reg_alpha`, `reg_lambda`, `min_child_weight`, `gamma`, `scale_pos_weight`

### 5. Stacking Ensemble
```
Layer 1 — Base Models:
  ├── XGBoost v1       (tuned — main anchor)
  ├── XGBoost v2       (different depth/colsample for diversity)
  ├── Random Forest    (tuned — bagging diversity)
  └── ExtraTrees       (tuned — random split diversity)

Layer 2 — Meta-Model:
  └── Logistic Regression (C=0.3) — learns optimal blend
```

Soft voting replaced with stacking so the meta-model learns *when* to trust each base model rather than blending with fixed weights.

### 6. Overfitting Check
Per-fold train vs validation AUC gap monitored across all 5 folds. Gap < 0.06 confirmed — no overfitting detected.

---

## 📈 Results

| Model | CV AUC-ROC |
|---|---|
| Logistic Regression (baseline) | 0.769 |
| Random Forest (tuned) | 0.783 |
| Gradient Boosting (baseline) | 0.782 |
| Soft Voting Ensemble (baseline) | 0.788 |
| **XGBoost + Stacking (final)** | **~0.78–0.79** |

> **Note:** Data ceiling analysis (intentionally overfit RF) confirmed the raw dataset's maximum separability is ~0.77 AUC, indicating label noise and missing behavioural signals are the primary limiting factors — not model choice.

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap
```

### Steps
```bash
# 1. Clone the repo
git clone https://github.com/your-username/streammax-fatigue-prediction.git
cd streammax-fatigue-prediction

# 2. Place data files in /data
#    ott_train.csv and ott_test.csv

# 3. Launch notebook
jupyter notebook TeamName_XGBoost_Pipeline_Final.ipynb

# 4. Run all cells — predictions saved automatically as:
#    TeamName_Predictions.csv
```

### Expected Runtime
| Section | Approx. Time |
|---|---|
| EDA & Feature Engineering | ~1 min |
| XGBoost Tuning (Stage 1, 100 iter) | ~15–25 min |
| Early Stopping | ~3–5 min |
| Model Comparison (RF + ET tuning) | ~10 min |
| Stacking + Passthrough experiment | ~10 min |
| Overfitting Check | ~3 min |
| SHAP Analysis | ~2 min |
| **Total** | **~45–60 min** |

---

## 📤 Output Format

`TeamName_Predictions.csv` — exactly 2,000 rows:

```
user_id,predicted_fatigue_probability
U000001,0.7823
U000002,0.1204
U000003,0.9341
...
```

Values between 0.0 (very engaged) and 1.0 (severely fatigued).

---

## 🎯 Business Strategy Summary

### Risk Segmentation
Users are bucketed into 4 tiers based on predicted probability:

| Segment | Probability | Recommended Action |
|---|---|---|
| 🔴 Critical Risk | > 0.75 | Immediate win-back: SMS + free upgrade offer within 24hrs |
| 🟠 High Risk | 0.50–0.75 | Day 3–5 inactivity: curated "Top Picks" notification |
| 🟡 Medium Risk | 0.30–0.50 | Weekly digest + "Continue Watching" nudges |
| 🟢 Low Risk | < 0.30 | Loyalty milestone rewards, upgrade prompts |

### Key Fatigue Signals (in order of importance)
1. **Days since last session** — Absence of 7+ days = 3.2× higher fatigue probability
2. **Viewing trend (7d vs 30d)** — Declining watch time is more predictive than absolute level
3. **Completion rate** — Low completion indicates content mismatch, not just low interest
4. **Recommendation CTR** — Algorithm drift signal; low CTR triggers taste re-survey
5. **Zero binge sessions** — Loss of deep engagement precedes cancellation

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-blue)
![Pandas](https://img.shields.io/badge/Pandas-latest-blue)
![SHAP](https://img.shields.io/badge/SHAP-explainability-green)

---

## 👥 Team

**Team Name** · StrategiX 2.0 · Marvix Analytics Hackathon · February 2026

---

## 📄 License

This project was developed for the StrategiX 2.0 competition. Data provided by the competition organizers — not for redistribution.
