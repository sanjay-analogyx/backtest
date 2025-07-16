import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool as MP_Pool, Manager
from pathlib import Path
from tqdm import tqdm
import re


def process_symbol_helper(args):
    return process_symbol(*args)


def init_process(shared_df, base_models, base_model_features, base_model_medians, model, features, median,
                 sorted_features):
    global df_global, base_models_global, base_model_features_global, base_model_medians_global
    global model_global, features_global, median_global, sorted_features_global

    df_global = shared_df
    base_models_global = base_models
    base_model_features_global = base_model_features
    base_model_medians_global = base_model_medians
    model_global = model
    features_global = features
    median_global = median
    sorted_features_global = sorted_features


def convert_numeric_to_range_buckets(df, n_bins=5, strategy='quantile', exclude_cols=None):
    """
    Converts numeric columns to categorical bins and drops the original columns.
    Also prints which columns were converted.
    """

    df = df.copy()
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    eligible_cols = [col for col in numeric_cols if col not in exclude_cols and col != 'trade_outcome']

    converted_cols = []

    for col in eligible_cols:
        try:
            if strategy == 'quantile':
                df[col + '_bin'] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
            else:
                df[col + '_bin'] = pd.cut(df[col], bins=n_bins, labels=False)
            df.drop(columns=col, inplace=True)
            converted_cols.append(col)
        except Exception as e:
            print(f"⚠️ Skipping binning for {col}: {e}")

    if converted_cols:
        print(f"✅ Converted and dropped {len(converted_cols)} columns: {converted_cols}")

    return df


# ✅ Function to Process Each Symbol in Parallel
def process_symbol(
        symbol,
        threshold_min,
        stop_loss_percent,
        target_percent,
        lot_quantity,
        atm_diff,
        breakeven_pct,
        breakeven_trail_pct,
        trail_trigger_pct,
        trail_buffer_pct,
):
    global df_global, base_models_global, base_model_features_global, base_model_medians_global
    global model_global, features_global, median_global, sorted_features_global

    # print(symbol)
    df_filtered = df_global[df_global["symbol"] == symbol].copy()
    df_filtered_original = df_filtered.copy()

    meta_feature_names = []
    # ✅ First predict using base models
    for model_name, base_model in base_models_global.items():
        feature_list = base_model_features_global[model_name]
        base_median = base_model_medians_global[model_name]

        df_selected = (
            df_filtered[feature_list]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(base_median)
        )
        pred_col = f'pred_{model_name}'
        df_filtered[pred_col] = base_model.predict_proba(df_selected)[:, 1]

    # --- Now build the exact meta‐model input using the model’s own feature list ---
    expected = model_global.booster_.feature_name()  # the 554 names/order it was trained on
    df_meta_input = (
        df_filtered[features_global]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(median_global)
            .reindex(columns=expected)
    )

    # sanity check
    assert list(df_meta_input.columns) == expected, (
        f"Model expects {len(expected)} features, "
        f"but got {df_meta_input.shape[1]}"
    )
    df_meta_for_output = df_meta_input.copy()
    df_meta_for_output.index = df_filtered["date"].values

    predictions = model_global.predict(df_meta_input)
    probabilities = model_global.predict_proba(df_meta_input)[:, 1]

    # ✅ Restore non-numeric columns
    df_filtered = df_filtered_original.copy()
    df_filtered.set_index("date", inplace=True)
    row_counter = 0
    MARKET_CLOSE_TIME = "15:30:00"

    trade_active = False
    last_exit_time = None
    trades = []
    output_data = []

    for (index, row), pred, prob in zip(df_filtered.iterrows(), predictions, probabilities):

        meta_row = df_meta_for_output.iloc[row_counter]
        row_counter += 1

        if index.time() > pd.to_datetime("15:30:00").time():
            continue

        confidence = round(prob * 100, 2)

        row_data = {
            "date": index,
            "prediction": pred,
            "confidence": prob,
        }

        row_data.update(meta_row.to_dict())

        # Add sorted features explicitly first
        for feature in sorted_features_global:
            row_data[f"feature_{feature}"] = row.get(feature, np.nan)

        # Finally, append original row data (will appear last)
        row_data.update(row.to_dict())

        output_data.append(row_data)

        if confidence > threshold_min and row["atm_diff"] <= atm_diff:

            if trade_active or (last_exit_time and index <= last_exit_time):
                continue

            entry_price = row["close"]
            entry_time = index

            stop_loss_price = entry_price * (1 - stop_loss_percent / 100)

            target_price = entry_price * (1 + target_percent / 100)

            trade_active = True

            future_data = df_filtered.loc[index:].copy()
            future_data = future_data[future_data.index > index]
            future_data = future_data[future_data.index.date == index.date()]
            future_data = future_data[future_data.index.time <= pd.to_datetime(MARKET_CLOSE_TIME).time()]

            exit_price = None
            exit_time = None
            exit_reason = None

            breakeven_multiplier = 1 + breakeven_pct / 100
            breakeven_trail_multiplier = 1 + breakeven_trail_pct / 100
            x_trail = trail_trigger_pct / 100
            y_trail = trail_buffer_pct / 100

            # ✅ New: Initialize trackers
            highest_high = entry_price
            lowest_low = entry_price
            time_of_highest_high = entry_time
            time_of_lowest_low = entry_time

            for future_index, future_row in future_data.iterrows():

                if future_row["low"] <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_time = future_index
                    if exit_reason is None:
                        exit_reason = "Stop-Loss Hit"
                    break

                elif future_row["high"] >= target_price and exit_reason != "Target Hit":
                    exit_reason = "Target Hit"
                    stop_loss_price = target_price

                elif future_row[
                    "high"] >= entry_price * breakeven_multiplier and stop_loss_price < entry_price * breakeven_trail_multiplier:
                    exit_reason = "BreakEven Hit"
                    stop_loss_price = entry_price * breakeven_trail_multiplier

                elif exit_reason == "Target Hit" and future_row["high"] >= stop_loss_price * (1 + x_trail):
                    stop_loss_price = future_row["high"] * (1 - y_trail)

            if exit_price is None:
                market_close_time = pd.to_datetime(MARKET_CLOSE_TIME).time()
                same_day_data = future_data[future_data.index.date == index.date()]
                final_candle = same_day_data[same_day_data.index.time == market_close_time]

                if not final_candle.empty:
                    exit_price = final_candle.iloc[0]["close"]
                    exit_time = final_candle.index[0]
                    exit_reason = "Time-based Exit at 3:25 PM"
                else:
                    eligible_candles = same_day_data[same_day_data.index.time < market_close_time]
                    if not eligible_candles.empty:
                        last_candle = eligible_candles.iloc[-1]
                        exit_price = last_candle["close"]
                        exit_time = eligible_candles.index[-1]
                        exit_reason = "Latest Available Exit Before 3:25 PM"
                    else:
                        exit_price = row["close"]
                        exit_time = index
                        exit_reason = "Exit due to missing same-day data"

            if exit_time:
                last_exit_time = exit_time

            trade_active = False

            qty = lot_quantity
            buying_capital = entry_price * qty
            selling_capital = exit_price * qty
            pl = selling_capital - buying_capital
            charges = abs(buying_capital) * 0.005
            effective_pl = pl - charges
            # trade_features = {feature: row.get(feature, np.nan) for feature in sorted_features_global}
            trades.append({
                "Entry Date": entry_time.date(),
                "Entry Time": entry_time.time(),
                "Exit Date": exit_time.date(),
                "Exit Time": exit_time.time(),
                "StrikePrice": row["strike"],
                "Symbol": symbol,
                "Option Type": row["option_type"],
                "Quantity": qty,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Highest High After Entry": highest_high,
                "Time of Highest High": time_of_highest_high,
                "Lowest Low After Entry": lowest_low,
                "Time of Lowest Low": time_of_lowest_low,
                "Buying Capital": buying_capital,
                "Selling Capital": selling_capital,
                "Stop Loss": stop_loss_price,
                "Target": target_price,
                "P/L": pl,
                "Charges(.5%)": charges,
                "Effective P/L": effective_pl,
                "Exit Reason": exit_reason,
                "Confidence": confidence,
                "Original Trade Outcome": row["trade_outcome"],
                "Original Exit Price": row["exit_price"],
                "Original Exit Time": row["exit_time"],
                # **trade_features,
            })
    df_output = pd.DataFrame(output_data)
    import datetime as dt
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_symbol = re.search('\D+', symbol)[0]
    Path(f'backtest_results/{main_symbol}').mkdir(exist_ok=True)
    df_output.to_csv(f"backtest_results/{main_symbol}/{symbol}_{timestamp}.csv", index=False)
    return trades


def get_sorted_features(model, features):
    importances = extract_feature_importances(model, features)
    if importances is not None:
        return importances.sort_values(ascending=False).index.tolist()
    return []


def extract_feature_importances(model, features):
    """
    Extract feature importances from models like XGBoost, LightGBM, CatBoost.
    For stacking models, averages importance across base estimators if needed.
    """
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=features)
    elif hasattr(model, 'estimators_'):
        try:
            importances_list = []
            for name, est in model.named_estimators_.items():
                if hasattr(est, 'feature_importances_'):
                    importances_list.append(pd.Series(est.feature_importances_, index=features))
            if importances_list:
                importances = pd.concat(importances_list, axis=1).mean(axis=1)
        except Exception as e:
            print(f"⚠️ Error extracting importances from base models: {str(e)}")
    return importances


def remove_overlapping_trades(trades_df):
    # trades_df = trades_df.sort_values(by=["Entry Date", "Entry Time"])
    ce_trades = trades_df[trades_df["Option Type"] == 0]
    pe_trades = trades_df[trades_df["Option Type"] == 1]

    def filter_non_overlap(df):
        result = []
        last_exit = None
        for _, row in df.iterrows():
            entry_dt = datetime.combine(row["Entry Date"], row["Entry Time"])
            exit_dt = datetime.combine(row["Exit Date"], row["Exit Time"])
            if last_exit is None or entry_dt > last_exit:
                result.append(row)
                last_exit = exit_dt
        return pd.DataFrame(result)

    ce_filtered = filter_non_overlap(ce_trades)
    pe_filtered = filter_non_overlap(pe_trades)
    return pd.concat([ce_filtered, pe_filtered]).sort_values(by=["Entry Date", "Entry Time"]).reset_index(drop=True)


def bt_main(
        df=None,
        lot_quantity=75,
        symbol='BHARATFORG',
        strike_diff=20,
        stop_loss_percent=10,
        target_percent=20,
        breakeven_pct=15,
        breakeven_trail_pct=1,
        trail_trigger_pct=5,
        trail_buffer_pct=2,
        ce_threshold_min=50,
        pe_threshold_min=50,
        model=None,
        cpu_cores=10
):
    df_global = None
    model_global = None
    features_global = None
    median_global = None
    sorted_features_global = None

    atm_diff = strike_diff / 2
    df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)
    model_bundle = model

    # ✅ Load meta models
    ce_meta_model = model_bundle["ce"]["meta_model"]
    ce_features = model_bundle["ce"]["features"]
    ce_median = model_bundle["ce"]["median"]

    pe_meta_model = model_bundle["pe"]["meta_model"]
    pe_features = model_bundle["pe"]["features"]
    pe_median = model_bundle["pe"]["median"]

    # ✅ Load base models
    # ✅ Reconstruct base models manually from saved structure
    ce_base_models = {
        "catboost": model_bundle["ce"]["base_catboost"]["model"],
        "lgbm": model_bundle["ce"]["base_lgbm"]["model"],
        "xgb": model_bundle["ce"]["base_xgb"]["model"],
    }
    ce_base_model_features = {
        "catboost": model_bundle["ce"]["base_catboost"]["features"],
        "lgbm": model_bundle["ce"]["base_lgbm"]["features"],
        "xgb": model_bundle["ce"]["base_xgb"]["features"],
    }
    ce_base_model_medians = {
        "catboost": model_bundle["ce"]["base_catboost"]["median"],
        "lgbm": model_bundle["ce"]["base_lgbm"]["median"],
        "xgb": model_bundle["ce"]["base_xgb"]["median"],
    }
    pe_base_models = {
        "catboost": model_bundle["pe"]["base_catboost"]["model"],
        "lgbm": model_bundle["pe"]["base_lgbm"]["model"],
        "xgb": model_bundle["pe"]["base_xgb"]["model"],
    }
    pe_base_model_features = {
        "catboost": model_bundle["pe"]["base_catboost"]["features"],
        "lgbm": model_bundle["pe"]["base_lgbm"]["features"],
        "xgb": model_bundle["pe"]["base_xgb"]["features"],
    }
    pe_base_model_medians = {
        "catboost": model_bundle["pe"]["base_catboost"]["median"],
        "lgbm": model_bundle["pe"]["base_lgbm"]["median"],
        "xgb": model_bundle["pe"]["base_xgb"]["median"],
    }

    ce_sorted_features = get_sorted_features(ce_meta_model, ce_features)
    pe_sorted_features = get_sorted_features(pe_meta_model, pe_features)

    all_trades = []

    with Manager() as manager:
        for meta_model, features, median, threshold_min, option_type, sorted_features, base_models, base_model_features, base_model_medians in [
            (ce_meta_model, ce_features, ce_median, ce_threshold_min, 0,
            ce_sorted_features, ce_base_models, ce_base_model_features, ce_base_model_medians),
            (pe_meta_model, pe_features, pe_median, pe_threshold_min, 1,
            pe_sorted_features, pe_base_models, pe_base_model_features, pe_base_model_medians)
        ]:
            df_subset = df[df["option_type"] == option_type].copy()
            symbols = df_subset["symbol"].unique().tolist()

            args_list = [
                (symbol, threshold_min, stop_loss_percent, target_percent, lot_quantity, atm_diff,
                breakeven_pct, breakeven_trail_pct, trail_trigger_pct, trail_buffer_pct)
                for symbol in symbols
            ]
            # res = []
            # for i in tqdm(args_list):
            #     init_process(df_subset, base_models, base_model_features, base_model_medians,
            #                 meta_model, features, median, sorted_features)
            #     res.append(process_symbol_helper(i))
                
            with MP_Pool(
                    processes=cpu_cores,
                    initializer=init_process,
                    initargs=(df_subset, base_models, base_model_features, base_model_medians,
                            meta_model, features, median, sorted_features)
            ) as pool:
                res = list(tqdm(
                    pool.imap_unordered(process_symbol_helper, args_list),
                    total=len(args_list),
                    desc=f"Processing {len(args_list)} symbols",
                    unit="symbol"
                ))
            all_trades.extend([trade for sublist in res for trade in sublist if trade])

    if all_trades:
        final_trades_df = pd.DataFrame(all_trades)
        final_trades_df.sort_values(by=["Entry Date", "Entry Time"], inplace=True)

        # After collecting all trades
        final_trades_df = remove_overlapping_trades(final_trades_df)

        final_trades_df.insert(0, "Index", range(1, len(final_trades_df) + 1))

        # Group by day and compute running total within each day
        final_trades_df['Continuous Winner/Lossers'] = final_trades_df.groupby(final_trades_df['Entry Date'])[
            'Original Trade Outcome'].cumsum()
        final_trades_df['Cumulative Return IntraDay'] = final_trades_df.groupby(final_trades_df['Entry Date'])[
            'Effective P/L'].cumsum()
        final_trades_df['Continuous Winner/Lossers Overall'] = final_trades_df['Original Trade Outcome'].cumsum()
        final_trades_df['Cumulative Return Overall'] = final_trades_df['Effective P/L'].cumsum()
        filename = f"{symbol}_trading_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        file_path = Path("backtest_results") / symbol / filename
        file_path.parent.mkdir(exist_ok=True)

        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            final_trades_df.to_excel(writer, sheet_name='Trades', index=False)
            pivot_table = final_trades_df.groupby("Entry Date").agg({
                "Option Type": "count", "Buying Capital": "mean",
                "P/L": "sum", "Effective P/L": "sum"
            }).reset_index().rename(columns={"Option Type": "No. Of Trades", "Buying Capital": "Avg Buying Capital"})
            pivot_table.to_excel(writer, sheet_name='Pivot Table', index=False)

        print(f"✅ Simulation complete! Predictions saved to {file_path}")
        return file_path
