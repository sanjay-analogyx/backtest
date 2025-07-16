from pathlib import Path

# 📊 Data Handling
import pandas as pd
import numpy as np

# 📈 Visualization
import matplotlib.pyplot as plt

# 🚀 Gradient Boosting Libraries
from xgboost import XGBClassifier

# 📊 Metrics
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop


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
        importances = pd.Series(model.feature_importances_, index=list(X_train.columns))
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


def train_and_evaluate(df, option_type, split_date, symbol):

    df = df[df['option_type'] == option_type]

    df = df.replace([np.inf, -np.inf], np.nan)
    median = df.median(numeric_only=True)
    df = df.fillna(median)

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # Time-based split
    split_date = pd.to_datetime(split_date)
    train_df = df[df['date'] < split_date]
    test_df = df[df['date'] >= split_date]

    print(f"\n📅 [{option_type}] Train Date Range: {train_df['date'].min()} → {train_df['date'].max()}")
    print(f"📅 [{option_type}] Test Date Range : {test_df['date'].min()} → {test_df['date'].max()}")

    # # Select numeric columns and drop the label column
    X_train = train_df.select_dtypes(include=[np.number]).drop(columns=['trade_outcome'], errors='ignore')
    y_train = train_df['trade_outcome']
    X_test = test_df.select_dtypes(include=[np.number]).drop(columns=['trade_outcome'], errors='ignore')
    y_test = test_df['trade_outcome']

    # Remove correlated features
    X_train, dropped = remove_highly_correlated_features(X_train, threshold=0.95)
    X_test = X_test.drop(columns=dropped, errors='ignore')

    best_xgb = XGBClassifier(
        colsample_bytree=0.6,
        learning_rate=0.2,
        max_depth=9,
        n_estimators=500,
        n_jobs=-1,
        objective='binary:logistic',
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=5.0,
        subsample=0.6,
        scale_pos_weight=5.0,
        enable_categorical=False,
        eval_metric='logloss',
        gamma=0.0
    )

    # 5) Hand off to your evaluator
    model, threshold = evaluate_model(
        best_xgb,
        X_train,
        y_train,
        X_test,
        y_test,
        f"{symbol}_xgb_{option_type}"
    )
    return model, list(X_train.columns), median, threshold


def xgb_main(df=None, symbol=None, split_date='2025-01-01'):

    df['trade_outcome'] = df['trade_outcome'].map({-1: 0, 1: 1})

    ce_model, ce_features, ce_median, ce_threshold = train_and_evaluate(df.copy(), 'CE', split_date, symbol)
    pe_model, pe_features, pe_median, pe_threshold = train_and_evaluate(df.copy(), 'PE', split_date, symbol)

    model_dict = {
        "ce": {
            "model": ce_model,
            "features": ce_features,
            "median": ce_median,
            "threshold": ce_threshold
        },
        "pe": {
            "model": pe_model,
            "features": pe_features,
            "median": pe_median,
            "threshold": pe_threshold
        }
    }
    return model_dict
