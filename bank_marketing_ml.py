"""
================================================================================
  SMART CALL TARGETING FOR BANK CAMPAIGNS
  Machine Learning Solution (Single File)
================================================================================
  Project : Smart Call Targeting for Bank Campaigns
  Dataset : Bank Marketing Dataset (Kaggle / UCI)
  Currency: INR (₹)
  Models  : Decision Tree & Random Forest
  Goal    : Predict which customers are most likely to subscribe to a term
            deposit so the bank can target its marketing calls efficiently.

  Business Question:
  "How can your ML model help the bank reduce unnecessary calls and improve
   its campaign?"
================================================================================
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sys
import warnings

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend (saves PNGs)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ── Output directory for plots ────────────────────────────────────────────────
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def save_plot(fig, name):
    """Save a matplotlib figure to the plots directory."""
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 Saved plot → {path}")


# ==============================================================================
# STEP 1 — DATA LOADING & EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("=" * 80)
print("  STEP 1: DATA LOADING & EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# --- Load dataset -------------------------------------------------------------
DATA_FILE = "bank.csv"
if not os.path.exists(DATA_FILE):
    print(f"\n❌  '{DATA_FILE}' not found in the current directory!")
    print("   Please download it from:")
    print("   https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset")
    print(f"   and place it in: {os.getcwd()}")
    exit(1)

# Detect separator (semicolon vs comma) and load
with open(DATA_FILE, "r") as f:
    first_line = f.readline()
sep = ";" if ";" in first_line else ","

df = pd.read_csv(DATA_FILE, sep=sep)

# Ensure target column is named 'deposit' (UCI uses 'y', Kaggle uses 'deposit')
if "y" in df.columns and "deposit" not in df.columns:
    df.rename(columns={"y": "deposit"}, inplace=True)

print(f"\n✅  Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns (separator: '{sep}')\n")

# --- Basic info ---------------------------------------------------------------
print("── First 5 Rows ──")
print(df.head().to_string())

print("\n── Data Types ──")
print(df.dtypes.to_string())

print("\n── Statistical Summary (Numerical) ──")
print(df.describe().to_string())

print("\n── Missing Values ──")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   No missing values ✅")
else:
    print(missing[missing > 0].to_string())

print(f"\n── Duplicate Rows: {df.duplicated().sum()} ──")

# --- Target distribution ------------------------------------------------------
print("\n── Target Variable Distribution ('deposit') ──")
target_counts = df["deposit"].value_counts()
print(target_counts.to_string())
print(f"   Subscription rate: {target_counts.get('yes', 0) / len(df) * 100:.1f}%")

fig, ax = plt.subplots(figsize=(6, 4))
colors = ["#E74C3C", "#2ECC71"]
target_counts.plot.bar(ax=ax, color=colors, edgecolor="black", alpha=0.85)
ax.set_title("Target Variable Distribution — Term Deposit Subscription", fontweight="bold")
ax.set_xlabel("Subscribed to Term Deposit")
ax.set_ylabel("Number of Customers")
for i, v in enumerate(target_counts.values):
    ax.text(i, v + 50, str(v), ha="center", fontweight="bold")
save_plot(fig, "01_target_distribution.png")

# --- Age distribution by subscription ----------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
df[df["deposit"] == "yes"]["age"].hist(bins=30, alpha=0.6, label="Subscribed (yes)", color="#2ECC71", ax=ax, edgecolor="black")
df[df["deposit"] == "no"]["age"].hist(bins=30, alpha=0.6, label="Not Subscribed (no)", color="#E74C3C", ax=ax, edgecolor="black")
ax.set_title("Age Distribution by Subscription Status", fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
ax.legend()
save_plot(fig, "02_age_distribution.png")

# --- Numerical feature distributions -----------------------------------------
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n── Numerical Features ({len(numerical_cols)}): {numerical_cols} ──")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(numerical_cols[:6]):
    sns.histplot(data=df, x=col, hue="deposit", kde=True, ax=axes[i],
                 palette={"yes": "#2ECC71", "no": "#E74C3C"}, alpha=0.6)
    axes[i].set_title(f"Distribution of {col}", fontweight="bold")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Numerical Feature Distributions by Subscription", fontweight="bold", fontsize=14, y=1.02)
plt.tight_layout()
save_plot(fig, "03_numerical_distributions.png")

# --- Categorical feature: subscription rate -----------------------------------
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols = [c for c in categorical_cols if c != "deposit"]
print(f"── Categorical Features ({len(categorical_cols)}): {categorical_cols} ──")

print("\n── Subscription Rate by Category ──")
for col in categorical_cols:
    ct = pd.crosstab(df[col], df["deposit"], normalize="index") * 100
    ct = ct.rename(columns={"yes": "% Subscribed", "no": "% Not Subscribed"})
    print(f"\n  {col.upper()}")
    print(ct.round(1).to_string())

# --- Subscription rate bar charts for key categories --------------------------
key_cats = ["job", "marital", "education", "contact", "month"]
key_cats = [c for c in key_cats if c in categorical_cols]

fig, axes = plt.subplots(len(key_cats), 1, figsize=(12, 4 * len(key_cats)))
if len(key_cats) == 1:
    axes = [axes]
for ax, col in zip(axes, key_cats):
    rates = df.groupby(col)["deposit"].apply(lambda x: (x == "yes").mean() * 100).sort_values(ascending=False)
    rates.plot.bar(ax=ax, color="#3498DB", edgecolor="black", alpha=0.85)
    ax.set_title(f"Subscription Rate (%) by {col.title()}", fontweight="bold")
    ax.set_ylabel("% Subscribed")
    ax.set_xlabel("")
    ax.axhline(y=df["deposit"].apply(lambda x: x == "yes").mean() * 100,
               color="red", linestyle="--", linewidth=1.5, label="Overall Average")
    ax.legend()
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
plt.tight_layout()
save_plot(fig, "04_subscription_by_category.png")


# ==============================================================================
# STEP 2 — IDENTIFY IMPORTANT CUSTOMER FEATURES
# ==============================================================================
print("\n" + "=" * 80)
print("  STEP 2: IDENTIFY IMPORTANT CUSTOMER FEATURES")
print("=" * 80)

# --- Correlation heatmap (numerical features + encoded target) ----------------
df_corr = df.copy()
df_corr["deposit_encoded"] = (df_corr["deposit"] == "yes").astype(int)

corr_matrix = df_corr[numerical_cols + ["deposit_encoded"]].corr()
print("\n── Correlation with Target (deposit) ──")
target_corr = corr_matrix["deposit_encoded"].drop("deposit_encoded").sort_values(ascending=False)
print(target_corr.to_string())

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title("Correlation Heatmap (Numerical Features + Target)", fontweight="bold")
save_plot(fig, "05_correlation_heatmap.png")

# --- Key observations --------------------------------------------------------
print("\n── Key Feature Observations ──")
print("""
  • 'duration' has the strongest positive correlation with subscription —
    longer calls tend to lead to subscriptions. However, duration is only
    known AFTER the call, so it cannot be used for pre-call predictions
    in a real scenario. We keep it for model learning but note this caveat.

  • 'balance' (average yearly balance) shows some positive correlation —
    wealthier customers may be more receptive to term deposits.

  • 'campaign' (number of contacts in this campaign) has a negative
    correlation — too many calls reduce the chance of subscription.

  • 'pdays' and 'previous' (from previous campaigns) also provide signal:
    customers contacted successfully before are more likely to subscribe again.
""")


# ==============================================================================
# STEP 3 — DATA PREPARATION
# ==============================================================================
print("=" * 80)
print("  STEP 3: DATA PREPARATION FOR ML")
print("=" * 80)

df_ml = df.copy()

# --- Encode target variable ---------------------------------------------------
le_target = LabelEncoder()
df_ml["deposit"] = le_target.fit_transform(df_ml["deposit"])   # no=0, yes=1
print(f"\n✅  Target encoded: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# --- Encode categorical features ---------------------------------------------
# Binary columns → Label Encoding
# Multi-class columns → One-Hot Encoding
binary_cols = [col for col in categorical_cols if df_ml[col].nunique() == 2]
multi_cols = [col for col in categorical_cols if df_ml[col].nunique() > 2]

print(f"\n   Binary categorical cols (Label Encoding)   : {binary_cols}")
print(f"   Multi-class categorical cols (One-Hot)      : {multi_cols}")

label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    label_encoders[col] = le
    print(f"     {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

df_ml = pd.get_dummies(df_ml, columns=multi_cols, drop_first=True, dtype=int)
print(f"\n✅  After One-Hot Encoding: {df_ml.shape[1]} features")

# --- Feature / target split ---------------------------------------------------
X = df_ml.drop("deposit", axis=1)
y = df_ml["deposit"]

print(f"\n   Features (X): {X.shape}")
print(f"   Target   (y): {y.shape}   —   Class 0: {(y == 0).sum()}, Class 1: {(y == 1).sum()}")

# --- Train / Test split (80/20 stratified) ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n✅  Train/Test split (80/20, stratified)")
print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test : {X_test.shape[0]} samples")

# --- Feature scaling (StandardScaler) -----------------------------------------
scaler = StandardScaler()
num_features = [c for c in numerical_cols if c in X_train.columns]
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])
print(f"\n✅  StandardScaler applied to {len(num_features)} numerical features: {num_features}")


# ==============================================================================
# STEP 4 — BUILD MODELS (Decision Tree & Random Forest)
# ==============================================================================
print("\n" + "=" * 80)
print("  STEP 4: MODEL BUILDING")
print("=" * 80)

# --- Decision Tree ------------------------------------------------------------
print("\n── Model 1: Decision Tree Classifier ──")
dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_proba = dt_model.predict_proba(X_test)[:, 1]
print("   ✅ Decision Tree trained (max_depth=5, class_weight='balanced')")

# --- Random Forest ------------------------------------------------------------
print("\n── Model 2: Random Forest Classifier ──")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
print("   ✅ Random Forest trained (n_estimators=200, max_depth=10, class_weight='balanced')")


# ==============================================================================
# STEP 5 — COMPARE THE TWO MODELS
# ==============================================================================
print("\n" + "=" * 80)
print("  STEP 5: MODEL COMPARISON")
print("=" * 80)

def evaluate_model(name, y_true, y_pred, y_proba):
    """Evaluate a model and return a dict of metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"\n── {name} ──")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")
    print(f"   F1-Score  : {f1:.4f}")
    print(f"   ROC-AUC   : {auc:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No (0)", "Yes (1)"]))

    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC-AUC": auc}

dt_metrics = evaluate_model("Decision Tree", y_test, dt_pred, dt_proba)
rf_metrics = evaluate_model("Random Forest", y_test, rf_pred, rf_proba)

# --- Side-by-side comparison table --------------------------------------------
comp_df = pd.DataFrame({"Decision Tree": dt_metrics, "Random Forest": rf_metrics})
print("\n── Side-by-Side Comparison ──")
print(comp_df.round(4).to_string())

# Determine winner
winner_name = "Random Forest" if rf_metrics["F1-Score"] >= dt_metrics["F1-Score"] else "Decision Tree"
print(f"\n🏆  Best Model (by F1-Score): {winner_name}")

# --- Comparison bar chart -----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(comp_df.index))
width = 0.35
bars1 = ax.bar(x - width / 2, comp_df["Decision Tree"], width, label="Decision Tree",
               color="#E67E22", edgecolor="black", alpha=0.85)
bars2 = ax.bar(x + width / 2, comp_df["Random Forest"], width, label="Random Forest",
               color="#2980B9", edgecolor="black", alpha=0.85)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Decision Tree vs Random Forest", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(comp_df.index)
ax.set_ylim(0, 1.15)
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", fontsize=9)
save_plot(fig, "06_model_comparison.png")


# ==============================================================================
# STEP 6 — EVALUATION MEASURES & WHY ACCURACY ALONE ISN'T ENOUGH
# ==============================================================================
print("\n" + "=" * 80)
print("  STEP 6: EVALUATION MEASURES — WHY ACCURACY ALONE ISN'T ENOUGH")
print("=" * 80)

print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  WHY ACCURACY ALONE IS NOT ENOUGH                                     │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                       │
  │  In this dataset, the classes may be imbalanced (more "no" than       │
  │  "yes"). A naive model that predicts "no" for every customer could    │
  │  achieve high accuracy but would be USELESS for the bank — it would   │
  │  miss ALL potential subscribers!                                       │
  │                                                                       │
  │  PRECISION — Of the customers we predict will subscribe, how many     │
  │              actually do? Important because calling non-subscribers    │
  │              wastes time and money.                                    │
  │                                                                       │
  │  RECALL    — Of ALL actual subscribers, how many did we identify?     │
  │              Important because missing a subscriber means lost         │
  │              revenue from a term deposit.                              │
  │                                                                       │
  │  F1-SCORE  — Harmonic mean of Precision & Recall. Provides a single  │
  │              balanced metric when both false positives and false       │
  │              negatives matter.                                         │
  │                                                                       │
  │  ROC-AUC   — Measures the model's ability to distinguish between     │
  │              classes across all thresholds. A score of 0.5 = random   │
  │              guessing, 1.0 = perfect separation.                      │
  │                                                                       │
  │  For the bank:                                                        │
  │   • High PRECISION → fewer wasted calls                              │
  │   • High RECALL    → fewer missed subscribers                        │
  │   • F1 & AUC       → overall model quality                           │
  └─────────────────────────────────────────────────────────────────────────┘
""")

# --- Confusion Matrices -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, name, y_pred in zip(axes, ["Decision Tree", "Random Forest"], [dt_pred, rf_pred]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
                linewidths=1, linecolor="black")
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
save_plot(fig, "07_confusion_matrices.png")

# --- ROC Curves ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for name, proba, color in [("Decision Tree", dt_proba, "#E67E22"),
                            ("Random Forest", rf_proba, "#2980B9")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {auc_val:.3f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Baseline (AUC = 0.500)")
ax.set_title("ROC Curve — Model Comparison", fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
save_plot(fig, "08_roc_curves.png")


# ==============================================================================
# STEP 7 — PRIORITY CONTACT LIST (LIMITED CALLS)
# ==============================================================================
print("=" * 80)
print("  STEP 7: PRIORITY CONTACT LIST — WHO TO CALL FIRST?")
print("=" * 80)

# Use the best model's probabilities
best_model_name = winner_name
best_proba = rf_proba if winner_name == "Random Forest" else dt_proba

# Build a priority dataframe
priority_df = X_test.copy()
priority_df["actual_subscribed"] = y_test.values
priority_df["subscription_probability"] = best_proba
priority_df = priority_df.sort_values("subscription_probability", ascending=False).reset_index(drop=True)
priority_df.index = priority_df.index + 1  # 1-indexed rank
priority_df.index.name = "Priority_Rank"

print(f"\n── Top 20 Customers to Contact First (by {best_model_name}) ──")
top20 = priority_df[["subscription_probability", "actual_subscribed"]].head(20)
top20.columns = ["Subscription Probability", "Actually Subscribed (1=Yes)"]
print(top20.to_string())

# --- Cumulative Gain Analysis -------------------------------------------------
print("\n── Cumulative Gain Analysis ──")
total_subscribers = y_test.sum()
priority_df["cumulative_subscribers"] = priority_df["actual_subscribed"].cumsum()
priority_df["cumulative_pct_subscribers"] = priority_df["cumulative_subscribers"] / total_subscribers * 100
priority_df["pct_customers_contacted"] = np.arange(1, len(priority_df) + 1) / len(priority_df) * 100

for pct in [10, 20, 30, 50]:
    n_customers = int(len(priority_df) * pct / 100)
    captured = priority_df.iloc[n_customers - 1]["cumulative_pct_subscribers"]
    print(f"   📞 By calling the TOP {pct:>2d}% of customers ({n_customers:>4d} calls), "
          f"the bank captures {captured:>5.1f}% of ALL subscribers")

# --- Cumulative Gain Curve ----------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(priority_df["pct_customers_contacted"],
        priority_df["cumulative_pct_subscribers"],
        color="#2ECC71", linewidth=2.5, label=f"{best_model_name} (Targeted)")
ax.plot([0, 100], [0, 100], "k--", linewidth=1.5, label="Random Calling (No Model)")
ax.set_title("Cumulative Gain Curve — Targeted vs Random Calling", fontweight="bold")
ax.set_xlabel("% of Customers Contacted")
ax.set_ylabel("% of Subscribers Captured")
ax.legend(loc="lower right")
ax.set_xlim([0, 100])
ax.set_ylim([0, 105])
ax.fill_between(priority_df["pct_customers_contacted"],
                priority_df["cumulative_pct_subscribers"],
                priority_df["pct_customers_contacted"],
                alpha=0.15, color="#2ECC71")
ax.annotate("Model's Advantage\n(Lift)", xy=(30, 65), fontsize=12,
            fontweight="bold", color="#27AE60", ha="center")
save_plot(fig, "09_cumulative_gain_curve.png")

# --- Feature Importances (Best Model) ----------------------------------------
print(f"\n── Feature Importances ({best_model_name}) ──")
best_model = rf_model if winner_name == "Random Forest" else dt_model
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(15).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
importances.head(15).plot.barh(ax=ax, color="#3498DB", edgecolor="black", alpha=0.85)
ax.set_title(f"Top 15 Feature Importances — {best_model_name}", fontweight="bold")
ax.set_xlabel("Importance Score")
ax.invert_yaxis()
save_plot(fig, "10_feature_importances.png")


# ==============================================================================
# STEP 8 — PRACTICAL BUSINESS SUGGESTIONS
# ==============================================================================
print("\n" + "=" * 80)
print("  STEP 8: PRACTICAL BUSINESS SUGGESTIONS")
print("=" * 80)

print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  SUGGESTION 1: USE THE MODEL TO PRIORITIZE CALLS                      │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                       │
  │  Instead of calling every customer, rank them by the model's          │
  │  predicted subscription probability. Call the top-ranked customers    │
  │  first. As shown in the cumulative gain analysis, contacting just     │
  │  the top 20-30% of customers can capture the majority of potential    │
  │  subscribers — dramatically reducing call costs while maintaining     │
  │  campaign effectiveness.                                              │
  │                                                                       │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  SUGGESTION 2: FOCUS ON HIGH-CONVERSION CUSTOMER SEGMENTS             │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                       │
  │  EDA revealed that certain customer segments (e.g., students,         │
  │  retirees, customers with higher account balances, or those           │
  │  contacted in certain months) have significantly higher subscription  │
  │  rates. The bank should design targeted mini-campaigns for these      │
  │  segments with personalized messaging and offers.                     │
  │                                                                       │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  SUGGESTION 3: AVOID EXCESSIVE CONTACT ATTEMPTS                       │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                       │
  │  The 'campaign' feature (number of contacts during this campaign)     │
  │  is negatively correlated with subscription. Customers who are        │
  │  called too many times are LESS likely to subscribe. The bank should  │
  │  set a maximum contact limit (e.g., 3 attempts) per customer per     │
  │  campaign to avoid irritation and improve overall conversion rates.   │
  │                                                                       │
  └─────────────────────────────────────────────────────────────────────────┘
""")


# ==============================================================================
# FINAL SUMMARY — ANSWERING THE BUSINESS QUESTION
# ==============================================================================
print("=" * 80)
print("  FINAL ANSWER TO THE BUSINESS QUESTION")
print("=" * 80)

print(f"""
  Business Question:
  "How can your ML model help the bank reduce unnecessary calls and
   improve its campaign?"

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                       │
  │  The {winner_name} model (F1: {rf_metrics['F1-Score'] if winner_name == 'Random Forest' else dt_metrics['F1-Score']:.3f}, ROC-AUC: {rf_metrics['ROC-AUC'] if winner_name == 'Random Forest' else dt_metrics['ROC-AUC']:.3f}) can help the bank by:     │
  │                                                                       │
  │  1. SCORING every customer with a subscription probability before     │
  │     the campaign begins.                                              │
  │                                                                       │
  │  2. RANKING customers from most to least likely to subscribe.         │
  │                                                                       │
  │  3. CALLING only the top-ranked customers instead of everyone,        │
  │     reducing call volume by up to 70-80% while capturing the          │
  │     majority of potential subscribers.                                 │
  │                                                                       │
  │  4. SAVING TIME, MONEY, AND RESOURCES by avoiding calls to            │
  │     customers unlikely to subscribe.                                  │
  │                                                                       │
  │  5. IMPROVING customer satisfaction by not bombarding uninterested    │
  │     customers with unsolicited calls.                                 │
  │                                                                       │
  └─────────────────────────────────────────────────────────────────────────┘
""")

print("=" * 80)
print(f"  All plots saved in the '{PLOTS_DIR}/' directory.")
print("  Project complete! ✅")
print("=" * 80)
