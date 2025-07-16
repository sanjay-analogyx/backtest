# ⚙️ Standard Libraries
from pathlib import Path

# 📊 Data Handling
import pandas as pd
import numpy as np
from pathlib import Path

# 📈 Visualization
# 📈 Visualization
import matplotlib

matplotlib.use('Agg')  # ← force non-interactive backend (no tkinter)
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# 📊 Metrics
from sklearn.metrics import (
    accuracy_score, classification_report, 
    f1_score, precision_score, recall_score,
    )


def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop


def check_feature_drift(train_df, test_df, alpha=0.001, min_effect_size=0.1):
    from scipy.stats import ks_2samp

    drift_features = []
    for col in train_df.columns:
        stat, p = ks_2samp(train_df[col], test_df[col])
        effect_size = abs(train_df[col].mean() - test_df[col].mean()) / (train_df[col].std() + 1e-8)
        if p < alpha and effect_size > min_effect_size:
            drift_features.append(col)
    return drift_features


def find_best_threshold(probs, y_true, metric='f1'):
    thresholds = np.linspace(0.1, 0.9, 100)
    best_threshold = 0.5
    best_score = 0
    for t in thresholds:
        preds = (probs > t).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        if metric == 'recall':
            score = recall
        elif metric == 'precision':
            score = precision
        else:
            score = f1

        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"🔧 Threshold tuning: max {metric}={best_score:.4f} at threshold={best_threshold:.2f}")
    return best_threshold, best_score


def plot_precision_recall_threshold(probs, y_true, model_name):
    from sklearn.metrics import precision_recall_curve
    import os
    from datetime import datetime

    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    thresholds = np.append(thresholds, 1)  # match length

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision, label='Precision')
    plt.plot(thresholds, recall, label='Recall')
    plt.axvline(x=0.25, color='red', linestyle='--', label='Current threshold (0.25)')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Precision and Recall vs Threshold\n{model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Correct save location
    os.makedirs('graph', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'graph/{model_name}_precision_recall_{ts}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved Precision-Recall Curve at: {save_path}")


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    output_dir = Path("model_outputs")
    output_dir.mkdir(exist_ok=True)

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    best_threshold, best_f1 = find_best_threshold(probs, y_test, metric='f1')  # or 'f1' or 'precision' or 'recall'
    y_pred = (probs > best_threshold).astype(int)

    plot_precision_recall_threshold(probs, y_test, model_name)

    print(f"📌 Total trades in test set: {len(y_test)}")
    print(f"📌 Actual winners in test set: {y_test.sum()}")
    print(f"📌 Predicted winners: {y_pred.sum()}")

    print(f"\n📊 {model_name} Results")
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"🎯 Best Threshold: {best_threshold:.2f} | F1: {best_f1:.4f}")
    print(classification_report(y_test, y_pred))

    # 🧠 Feature Importance
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
    elif hasattr(model, 'estimators_'):
        try:
            importances_list = []
            for name, est in model.named_estimators_.items():
                if hasattr(est, 'feature_importances_'):
                    importances_list.append(pd.Series(est.feature_importances_, index=X_train.columns))
            if importances_list:
                importances = pd.concat(importances_list, axis=1).mean(axis=1)
        except Exception as e:
            print("⚠️ Error extracting importances from base models:", str(e))

    if importances is not None:
        top_features = importances.sort_values(ascending=False).head(300)
        print("\n🔍 Top 300 Feature Importances:")
        print(top_features.to_string())
    else:
        print("⚠️ Could not compute feature importances for this model.")

    return model, best_threshold


def tune_and_evaluate(model, param_dist, X_train, y_train, X_test, y_test, name):
    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=100,
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    search.fit(X_train, y_train)
    print(f"🔎 Best {name} params: {search.best_params_}")
    best = search.best_estimator_
    return best


def create_date_series(start_date, end_date):
    """
    Create a pandas date range between start_date and end_date (inclusive).
    Dates should be in 'YYYY-MM-DD' format.
    """
    return pd.date_range(start=start_date, end=end_date).date


def get_last_thursday_of_month(year: int, month: int) -> pd.Timestamp:
    """
    Returns the date of the last Thursday of the given month and year.
    """
    # Get last day of the month
    last_day = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    # Go backwards to the last Thursday
    offset = (last_day.weekday() - 3) % 7  # 3 is Thursday
    last_thursday = last_day - pd.Timedelta(days=offset)
    return last_thursday
