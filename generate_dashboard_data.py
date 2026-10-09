"""
Generate dashboard_data.json containing all analytical metrics, charts data,
priority lead lists, and model comparisons for the web dashboard.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Load dataset
sep = ";"
with open("bank.csv", "r") as f:
    if ";" not in f.readline():
        sep = ","

df = pd.read_csv("bank.csv", sep=sep)
if "y" in df.columns and "deposit" not in df.columns:
    df.rename(columns={"y": "deposit"}, inplace=True)

# Overview metrics
total_customers = int(len(df))
subscribers = int((df["deposit"] == "yes").sum())
non_subscribers = total_customers - subscribers
subscription_rate = round((subscribers / total_customers) * 100, 2)
avg_balance = round(float(df["balance"].mean()), 2)
avg_duration = round(float(df["duration"].mean()), 1)

# Demographic distributions
# Age distribution (histogram buckets)
bins = list(range(18, 90, 6))
labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
age_summary = df.groupby(["age_group", "deposit"], observed=False).size().unstack(fill_value=0)
age_chart = {
    "labels": [str(x) for x in age_summary.index],
    "yes": [int(x) for x in age_summary.get("yes", pd.Series(0, index=age_summary.index))],
    "no": [int(x) for x in age_summary.get("no", pd.Series(0, index=age_summary.index))]
}

# Conversion by categorical features
def get_cat_conversion(col):
    grp = df.groupby(col)["deposit"].agg(
        total="count",
        yes=lambda x: (x == "yes").sum()
    )
    grp["conversion_rate"] = (grp["yes"] / grp["total"]) * 100
    grp = grp.sort_values(by="total", ascending=False)
    return {
        "categories": list(grp.index.astype(str)),
        "totals": [int(x) for x in grp["total"]],
        "yes_count": [int(x) for x in grp["yes"]],
        "conversion_rates": [round(float(x), 2) for x in grp["conversion_rate"]]
    }

job_chart = get_cat_conversion("job")
education_chart = get_cat_conversion("education")
marital_chart = get_cat_conversion("marital")
month_chart = get_cat_conversion("month")

# Data preparation for ML
df_ml = df.drop(columns=["age_group"]).copy()
categorical_cols = df_ml.select_dtypes(include=["object"]).columns.tolist()
categorical_cols = [c for c in categorical_cols if c != "deposit"]

df_ml["deposit"] = (df_ml["deposit"] == "yes").astype(int)

binary_cols = [c for c in categorical_cols if df_ml[c].nunique() == 2]
multi_cols = [c for c in categorical_cols if df_ml[c].nunique() > 2]

for col in binary_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])

df_ml = pd.get_dummies(df_ml, columns=multi_cols, drop_first=True, dtype=int)

X = df_ml.drop("deposit", axis=1)
y = df_ml["deposit"]
feature_names = list(X.columns)

numerical_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
num_features = [c for c in numerical_cols if c in X.columns]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Keep unscaled test features for readable lead table display
raw_test_data = df.iloc[X_test.index].copy().reset_index(drop=True)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
X_test_scaled[num_features] = scaler.transform(X_test[num_features])

# Train Decision Tree
dt = DecisionTreeClassifier(
    max_depth=5, min_samples_split=10, min_samples_leaf=5,
    random_state=42, class_weight="balanced"
)
dt.fit(X_train_scaled, y_train)
dt_preds = dt.predict(X_test_scaled)
dt_proba = dt.predict_proba(X_test_scaled)[:, 1]

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=5,
    random_state=42, class_weight="balanced", n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]

def get_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    # downsample curve points to keep JSON compact
    indices = np.linspace(0, len(fpr) - 1, 40).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr[indices]],
            "tpr": [round(float(x), 4) for x in tpr[indices]]
        }
    }

dt_eval = get_metrics(y_test, dt_preds, dt_proba)
rf_eval = get_metrics(y_test, rf_preds, rf_proba)

# Feature importance from Random Forest
feat_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)
feature_importance = {
    "features": list(feat_imp.index),
    "scores": [round(float(x), 4) for x in feat_imp.values]
}

# Cumulative Gain curve
priority_indices = np.argsort(-rf_proba)
total_actual_yes = int(y_test.sum())
actual_sorted = y_test.iloc[priority_indices].values

pct_contacted = []
pct_captured = []
for p in [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]:
    cnt = int(len(actual_sorted) * (p / 100.0))
    captured = actual_sorted[:cnt].sum()
    pct_contacted.append(p)
    pct_captured.append(round((captured / total_actual_yes) * 100, 2))

cumulative_gain = {
    "pct_contacted": pct_contacted,
    "pct_captured": pct_captured,
    "baseline": pct_contacted
}

# Priority Leads list (top 50 leads to contact first)
priority_leads = []
for rank, idx in enumerate(priority_indices[:50], 1):
    row = raw_test_data.iloc[idx]
    prob = round(float(rf_proba[idx]) * 100, 1)
    priority_leads.append({
        "rank": rank,
        "age": int(row["age"]),
        "job": str(row["job"]),
        "marital": str(row["marital"]),
        "education": str(row["education"]),
        "balance": int(row["balance"]),
        "housing": str(row["housing"]),
        "loan": str(row["loan"]),
        "duration": int(row["duration"]),
        "campaign": int(row["campaign"]),
        "probability": prob,
        "actual": "yes" if y_test.iloc[idx] == 1 else "no"
    })

payload = {
    "overview": {
        "total_customers": total_customers,
        "subscribers": subscribers,
        "non_subscribers": non_subscribers,
        "subscription_rate": subscription_rate,
        "avg_balance": avg_balance,
        "avg_duration": avg_duration
    },
    "charts": {
        "age_chart": age_chart,
        "job_chart": job_chart,
        "education_chart": education_chart,
        "marital_chart": marital_chart,
        "month_chart": month_chart,
        "feature_importance": feature_importance,
        "cumulative_gain": cumulative_gain
    },
    "models": {
        "decision_tree": dt_eval,
        "random_forest": rf_eval
    },
    "priority_leads": priority_leads
}

with open("dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print("Generated dashboard_data.json successfully!")
