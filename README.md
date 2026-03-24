EX NO 1

# ============================================================
#  Customer Churn Prediction - Main Code
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# Reads the Telco CSV into a DataFrame and prints its shape
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ─────────────────────────────────────────────────────────────
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# - Shows first 5 rows to understand structure
# - Checks missing values so we know what to clean
# - Plots churn count (how many stayed vs left)
# - Plots tenure boxplot (do long-term customers churn less?)
# ─────────────────────────────────────────────────────────────
print(df.head())
print("Missing values:\n", df.isnull().sum())
print("Churn counts:\n", df["Churn"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["Churn"].value_counts().plot(kind="bar", ax=axes[0], color=["#4CAF50", "#F44336"])
axes[0].set_title("Churn Distribution")          # Green = stayed, Red = churned
df.boxplot(column="tenure", by="Churn", ax=axes[1])
axes[1].set_title("Tenure by Churn")             # Lower tenure = higher churn risk
plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────
# STEP 3: PREPROCESSING
# - Drop customerID (just an ID, no prediction value)
# - Fix TotalCharges: it has spaces stored as strings → convert to float
# - Fill missing TotalCharges with column median
# - Encode target: "Yes" → 1, "No" → 0
# - Label-encode binary Yes/No columns (e.g., PhoneService, PaperlessBilling)
# - One-hot encode multi-class columns (e.g., InternetService, Contract)
# ─────────────────────────────────────────────────────────────
df.drop(columns=["customerID"], inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # spaces → NaN
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)      # fill NaN

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})  # binary target

binary_cols = [c for c in df.columns if df[c].dtype == object and df[c].nunique() == 2]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])   # Male/Female → 0/1, Yes/No → 0/1

df = pd.get_dummies(df, drop_first=True)  # one-hot for Contract, InternetService etc.

X = df.drop(columns=["Churn"])  # features
y = df["Churn"]                 # target label
print(f"Features after encoding: {X.shape[1]}")

# ─────────────────────────────────────────────────────────────
# STEP 4: TRAIN/TEST SPLIT
# - 80% training, 20% testing
# - stratify=y keeps churn ratio equal in both splits
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────────────────────
# STEP 5: TRAIN 3 MODELS & EVALUATE
# Model 1 - Logistic Regression: simple, fast baseline (needs scaling)
# Model 2 - Random Forest: ensemble of 200 decision trees, handles non-linearity
# Model 3 - Gradient Boosting: boosted trees, usually highest accuracy
# For each model we print: precision, recall, f1-score, and ROC-AUC score
# ─────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),                           # scale features first
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1            # 200 trees, parallel
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, random_state=42  # slow learning = better
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]   # probability of churn (class 1)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n── {name} ──")
    print(classification_report(y_test, y_pred))  # precision, recall, f1
    print(f"ROC-AUC: {auc:.4f}")                  # 1.0 = perfect, 0.5 = random

    results[name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob, "auc": auc}

# ─────────────────────────────────────────────────────────────
# STEP 6: VISUALIZE RESULTS
# Plot 1 - ROC Curves: shows true positive vs false positive tradeoff per model
# Plot 2 - Confusion Matrix: actual vs predicted (TP, TN, FP, FN) for best model
# Plot 3 - Feature Importance: top 10 features driving churn prediction
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ["#2196F3", "#FF9800", "#9C27B0"]

# ROC Curves (higher curve = better model)
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=color, lw=2)
axes[0].plot([0, 1], [0, 1], "k--", lw=1)   # random baseline
axes[0].set_title("ROC Curves")
axes[0].legend()

# Confusion Matrix for best model (highest AUC)
best_name = max(results, key=lambda k: results[k]["auc"])
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(ax=axes[1], colorbar=False)
axes[1].set_title(f"Confusion Matrix ({best_name})")

# Feature Importance from Gradient Boosting
importances = results["Gradient Boosting"]["model"].feature_importances_
top_idx = np.argsort(importances)[-10:]      # top 10 most important features
axes[2].barh(X_train.columns[top_idx], importances[top_idx], color="#4CAF50")
axes[2].set_title("Top 10 Features (Gradient Boosting)")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("model_results.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────
# STEP 7: HYPERPARAMETER TUNING
# - Uses GridSearchCV to test all combinations of:
#   n_estimators (100 or 200 trees)
#   learning_rate (0.05 or 0.1 — how fast it learns)
#   max_depth (3 or 4 — how deep each tree grows)
# - 3-fold cross-validation, scored by ROC-AUC
# - Picks the combo with highest average AUC
# ─────────────────────────────────────────────────────────────
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4],
}
gs = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
)
gs.fit(X_train, y_train)
print(f"\nBest Params : {gs.best_params_}")
print(f"Best CV AUC : {gs.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 8: FINAL EVALUATION with tuned model
# Retest on the held-out test set using the best found parameters
# ─────────────────────────────────────────────────────────────
best_model = gs.best_estimator_
y_pred_tuned = best_model.predict(X_test)
y_prob_tuned = best_model.predict_proba(X_test)[:, 1]

print("\n── Final Tuned Model Performance ──")
print(classification_report(y_test, y_pred_tuned))
print(f"ROC-AUC (tuned): {roc_auc_score(y_test, y_prob_tuned):.4f}")
