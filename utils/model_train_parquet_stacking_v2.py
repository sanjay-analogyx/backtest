# 📊 Data Handling
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone

from lightgbm import LGBMClassifier

# 📊 Metrics
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


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


def train_and_evaluate_stacking(df, option_type, base_models, split_date, symbol):
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

    X_train = train_df.select_dtypes(include=[np.number]).drop(columns=['trade_outcome'], errors='ignore')
    y_train = train_df['trade_outcome']
    X_test = test_df.select_dtypes(include=[np.number]).drop(columns=['trade_outcome'], errors='ignore')
    y_test = test_df['trade_outcome']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = {name: np.zeros(len(X_train)) for name in base_models}
    test_preds = {name: np.zeros(len(X_test)) for name in base_models}

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        for name, model in base_models.items():
            cloned_model = clone(model)
            cloned_model.fit(X_tr, y_tr)
            oof_preds[name][val_idx] = cloned_model.predict_proba(X_val)[:, 1]
            test_preds[name] += cloned_model.predict_proba(X_test)[:, 1] / skf.n_splits

    # Add base model predictions as features
    for name in base_models.keys():
        X_train[f'pred_{name}'] = oof_preds[name]
        X_test[f'pred_{name}'] = test_preds[name]

    meta_model = LGBMClassifier(
        boosting_type='gbdt',
        colsample_bytree=0.7,
        importance_type='split',
        learning_rate=0.03,
        max_depth=7,
        min_child_samples=20,
        min_child_weight=0.001,
        min_split_gain=0.0,
        n_estimators=700,
        n_jobs=-1,
        num_leaves=31,
        objective='binary',
        random_state=42,
        reg_alpha=0.0,
        reg_lambda=0.0,
        subsample=0.8,
        subsample_for_bin=200000,
        subsample_freq=0,
        verbose=-1
    )

    model, threshold = evaluate_model(
        meta_model,
        X_train, y_train,
        X_test, y_test,
        model_name=f"{symbol}_stacked_lgbm_{option_type}"
    )

    return model, list(X_train.columns), median, threshold


def stacking_main(
        catboost_all=None,
        lgbm_all=None,
        xgb_all=None,
        df=None,
        symbol=None,
        split_date='2025-05-22',
        model_name=None
):
    df['trade_outcome'] = df['trade_outcome'].map({-1: 0, 1: 1})

    final_bundle = {}

    for option_type in ['CE', 'PE']:
        print(f"\n🚀 Training for {option_type}...")

        # Prepare base models dict
        base_models = {
            'catboost': catboost_all[option_type.lower()]['model'],
            'catboost_median': catboost_all[option_type.lower()]['median'],
            'catboost_features': catboost_all[option_type.lower()]['features'],

            'lgbm': lgbm_all[option_type.lower()]['model'],
            'lgbm_median': lgbm_all[option_type.lower()]['median'],
            'lgbm_features': lgbm_all[option_type.lower()]['features'],

            'xgb': xgb_all[option_type.lower()]['model'],
            'xgb_median': xgb_all[option_type.lower()]['median'],
            'xgb_features': xgb_all[option_type.lower()]['features'],

        }

        model_dict = {
            'catboost': catboost_all[option_type.lower()]['model'],
            'lgbm': lgbm_all[option_type.lower()]['model'],
            'xgb': xgb_all[option_type.lower()]['model'],
        }

        # Train stacked model
        meta_model, meta_features, meta_median, meta_threshold = train_and_evaluate_stacking(
            df.copy(), option_type, model_dict, split_date, symbol
        )

        final_bundle[option_type.lower()] = {
            "base_catboost": {
                "model": base_models['catboost'],
                "features": base_models['catboost_features'],
                "median": base_models['catboost_median']
            },
            "base_lgbm": {
                "model": base_models['lgbm'],
                "features": base_models['lgbm_features'],
                "median": base_models['lgbm_median']
            },
            "base_xgb": {
                "model": base_models['xgb'],
                "features": base_models['xgb_features'],
                "median": base_models['xgb_median']
            },
            "meta_model": meta_model,
            "features": meta_features,
            "median": meta_median,
            "threshold": meta_threshold
        }
    output_dir = Path(f"model_outputs")
    output_dir.mkdir(exist_ok=True)

    joblib.dump(final_bundle, output_dir / model_name.lower().replace("{symbol}", symbol))
    print(f"\n✅ All models saved together at: {output_dir / model_name.lower().replace('{symbol}', symbol)}")

    return final_bundle
