import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

from multiprocessing import Pool

import pandas as pd
import numpy as np
import ta
from tqdm import tqdm
import logging
import warnings
import py_vollib.black_scholes.implied_volatility as iv
from scipy.stats import norm
import os
from datetime import datetime, timedelta, time
# from indicator_calculation import indicator_main

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Suppress numpy runtime warnings like divide-by-zero, invalid sqrt, etc.
np.seterr(all='ignore')

# Optional: suppress pandas warnings if needed
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="py_lets_be_rational")


# ✅ Define Start Date for Data Selection
def label_trades(df, stop_loss_percent, target_percent):
    df = df.sort_values("date").reset_index(drop=True)
    for i, (stop_loss_percent, target_percent) in enumerate([(stop_loss_percent, target_percent)], start=1):
        df[f'stop_loss_price'] = df['close'] * (1 - stop_loss_percent/100)
        df[f'target_price'] = df['close'] * (1 + target_percent/100)

        outcomes = np.zeros(len(df))
        exit_prices = np.full(len(df), np.nan)
        exit_times = np.array([pd.NaT] * len(df))

        grouped = df.groupby('date_only', group_keys=False)

        for day, day_df in grouped:
            close = day_df['close'].values
            dates = day_df['date'].values
            stop_loss = day_df[f'stop_loss_price'].values
            target = day_df[f'target_price'].values

            for idx in range(len(day_df)):
                # Future candles for the SAME day
                future_close = close[idx + 1:]
                future_dates = dates[idx + 1:]

                if len(future_close) == 0:
                    continue
                future_high = day_df['high'].values[idx + 1:]
                future_low = day_df['low'].values[idx + 1:]

                target_hits = np.where(future_high >= target[idx])[0]
                sl_hits = np.where(future_low <= stop_loss[idx])[0]

                if len(target_hits) == 0 and len(sl_hits) == 0:
                    # ✅ Neither target nor stop loss hit, exit at EOD close
                    outcomes[day_df.index[idx]] = 0
                    exit_prices[day_df.index[idx]] = future_close[-1]
                    exit_times[day_df.index[idx]] = future_dates[-1]
                    continue

                first_target = target_hits[0] if len(target_hits) > 0 else np.inf
                first_sl = sl_hits[0] if len(sl_hits) > 0 else np.inf

                if first_target < first_sl:
                    outcomes[day_df.index[idx]] = 1
                    exit_prices[day_df.index[idx]] = target[idx]
                    exit_times[day_df.index[idx]] = future_dates[first_target]
                elif first_sl < first_target:
                    outcomes[day_df.index[idx]] = -1
                    exit_prices[day_df.index[idx]] = stop_loss[idx]
                    exit_times[day_df.index[idx]] = future_dates[first_sl]

        df[f'trade_outcome'] = outcomes
        df[f'exit_price'] = exit_prices
        df[f'exit_time'] = exit_times

    return df


def supertrend(df, period=10, multiplier=3):
    if len(df) < period:
        df['supertrend'] = np.nan
        df['lower_band'] = np.nan
        df['upper_band'] = np.nan
        return df

    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], period).average_true_range()
    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = np.ones(len(df))

    for i in range(1, len(df)):
        supertrend[i] = supertrend[i - 1]
        if df['close'].iloc[i] > upper_band.iloc[i - 1]:
            supertrend[i] = 1
        elif df['close'].iloc[i] < lower_band.iloc[i - 1]:
            supertrend[i] = 0
        else:
            supertrend[i] = supertrend[i - 1]
        if supertrend[i] == 1:
            lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i - 1])
        else:
            upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i - 1])

    df['supertrend'] = supertrend
    df['lower_band'] = lower_band
    df['upper_band'] = upper_band
    return df


def calculate_fibonacci(df, lookback=20):
    fib_high = df["high"].rolling(lookback).max()
    fib_low = df["low"].rolling(lookback).min()
    diff = fib_high - fib_low

    return df.assign(
        fib_high=fib_high,
        fib_low=fib_low,
        fib_23_6=fib_high - diff * 0.236,
        fib_38_2=fib_high - diff * 0.382,
        fib_50=fib_high - diff * 0.5,
        fib_61_8=fib_high - diff * 0.618,
        fib_78_6=fib_high - diff * 0.786
    )


def detect_engulfing(open_series, close_series):
    prev_open = open_series.shift(1)
    prev_close = close_series.shift(1)

    # Bullish Engulfing: Green candle engulfs previous red candle
    bullish_engulfing = (
            (prev_close < prev_open) &  # Previous red
            (close_series > open_series) &  # Current green
            (open_series < prev_close) &
            (close_series > prev_open)
    ).astype(int)

    # Bearish Engulfing: Red candle engulfs previous green candle
    bearish_engulfing = (
            (prev_close > prev_open) &  # Previous green
            (close_series < open_series) &  # Current red
            (open_series > prev_close) &
            (close_series < prev_open)
    ).astype(int)

    return bullish_engulfing, bearish_engulfing


def hammer(open, high, low, close):
    upper_wick = high - np.maximum(open, close)
    lower_wick = np.minimum(open, close) - low

    body = abs(close - open)
    candle_length = high - low

    is_hammer = (
            (lower_wick > 2 * body) &
            (upper_wick < 0.1 * body)
    )
    return is_hammer.astype(int)


def safe_rolling(group, window=20):
    return group.rolling(window, min_periods=1).mean()


def calculate_iv(row, r=0.06):
    try:
        return iv.implied_volatility(
            price=row['close'],
            S=row['underlying_close'],
            K=row['strike'],
            t=row['days_to_expiry'] / 365,
            r=r,
            flag='c' if row['option_type'] == 'CE' else 'p'
        )
    except Exception:
        return np.nan


# ✅ **Feature Engineering: Compute Technical Indicators**

def calculate_features(df):
    df["expiry_date"] = pd.to_datetime(df["expiry_date"].astype(str) + " 15:30:00")
    df["days_to_expiry"] = (df["expiry_date"] - df["date"]).dt.days
    df["minutes_to_expiry"] = (df["expiry_date"] - df["date"]).dt.total_seconds() / 60
    df.drop(columns=['expiry_date'], inplace=True)

    df['rolling_range'] = df['high'].rolling(20).max() - df['low'].rolling(20).min()
    df['green_streak'] = (df['close'] > df['open']).astype(int)
    df['red_streak'] = (df['close'] < df['open']).astype(int)
    df['volatility'] = df['close'].rolling(20).std()
    df["moneyness"] = np.log(df["underlying_close"] / df["strike"])  # log-moneyness is more standard
    df["delta"] = np.where(df["option_type"] == "CE", df["moneyness"], -df["moneyness"])
    date_grp = df.groupby(df['date_only'])
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.weekday
    df["atm_diff"] = abs(df["strike"] - df["underlying_close"])
    if len(df) >= 10:
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=10)
        df["underlying_atr"] = ta.volatility.average_true_range(df["underlying_high"], df["underlying_low"],
                                                                df["underlying_close"],
                                                                window=10)
    else:
        df["atr"] = np.nan
        df["underlying_atr"] = np.nan
    try:
        if len(df) >= 14:
            df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
            df["u_adx"] = ta.trend.adx(df["underlying_high"], df["underlying_low"], df["underlying_close"],
                                       window=14)
        else:
            df["adx"] = np.nan
            df["u_adx"] = np.nan
    except Exception as e:
        df["adx"] = np.nan
        df["underlying_adx"] = np.nan

    # ✅ VWAP
    df["vwap"] = ta.volume.volume_weighted_average_price(df["high"], df["low"], df["close"], df["volume"],
                                                        window=14)
    df["underlying_vwap"] = ta.volume.volume_weighted_average_price(df["underlying_high"], df["underlying_low"],
                                                                    df["underlying_close"], df["underlying_volume"],
                                                                    window=14)
    # ✅ MACD
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # ✅ RSI & Divergence
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["underlying_rsi"] = ta.momentum.rsi(df["underlying_close"], window=14)

    df["rsi_divergence"] = np.where(
        (df["close"] > df["close"].shift(14)) & (df["rsi"] < df["rsi"].shift(14)), 0,
        np.where((df["close"] < df["close"].shift(14)) & (df["rsi"] > df["rsi"].shift(14)), 1, 0)
    )

    df["underlying_rsi_divergence"] = np.where(
        (df["underlying_close"] > df["underlying_close"].shift(14)) & (
                df["underlying_rsi"] < df["underlying_rsi"].shift(14)), 0,
        np.where(
            (df["underlying_close"] < df["underlying_close"].shift(14)) & (
                    df["underlying_rsi"] > df["underlying_rsi"].shift(14)),
            1, 0)
    )

    # ✅ Bollinger Bands
    df["bollinger_high"] = ta.volatility.bollinger_hband(df["close"], window=20)
    df["bollinger_mid"] = ta.volatility.bollinger_mavg(df["close"], window=20)
    df["bollinger_low"] = ta.volatility.bollinger_lband(df["close"], window=20)

    # ✅ Bollinger Bands
    df["underlying_bollinger_high"] = ta.volatility.bollinger_hband(df["underlying_close"], window=20)
    df["underlying_bollinger_mid"] = ta.volatility.bollinger_mavg(df["underlying_close"], window=20)
    df["underlying_bollinger_low"] = ta.volatility.bollinger_lband(df["underlying_close"], window=20)

    # ✅ Moving Averages
    df["sma_5"] = ta.trend.sma_indicator(df["close"], window=5)
    df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["ema_9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["ema_21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["ema_crossover"] = np.where(df["ema_9"] > df["ema_21"], 1, 0)
    from ta.trend import sma_indicator
    # ✅ Moving Averages (SMA & EMA)
    df["underlying_sma_50"] = ta.trend.sma_indicator(df["underlying_close"], window=50)
    df["underlying_sma_200"] = ta.trend.sma_indicator(df["underlying_close"], window=200)
    df["underlying_sma_crossover"] = np.where(df["underlying_sma_50"] > df["underlying_sma_200"], 1, 0)

    # ✅ Option-Specific Features
    df["intrinsic_value"] = np.where(df["option_type"] == "CE",
                                    np.maximum(0, df["underlying_close"] - df["strike"]),
                                    np.maximum(0, df["strike"] - df["underlying_close"]))

    df["gamma"] = date_grp["delta"].diff().abs()
    df["theta"] = np.where(df["days_to_expiry"] > 0, -df["delta"] / df["days_to_expiry"], 0)

    # ✅ VIX Features
    df["vix_change"] = df["vix_close"].pct_change()
    df["vix_volatility"] = df["vix_close"].rolling(window=10).std()
    df["vix_sma_50"] = ta.trend.sma_indicator(df["vix_close"], window=50)
    df["vix_sma_200"] = ta.trend.sma_indicator(df["vix_close"], window=200)
    df["vix_sma_crossover"] = np.where(df["vix_sma_50"] > df["vix_sma_200"], 1, 0)

    # ✅ Feature Engineering
    df["rsi_slope"] = df["rsi"].diff()  # RSI Momentum
    df["adx_slope"] = df["adx"].diff()  # ADX Momentum
    df["bollinger_bandwidth"] = (df["bollinger_high"] - df["bollinger_low"]) / df["bollinger_mid"]

    # ✅ Lag Features
    for lag in range(1, 6):
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"rsi_lag_{lag}"] = df["rsi"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
        df[f"vwap_lag_{lag}"] = df["vwap"].shift(lag)
        df[f"delta_lag_{lag}"] = df["delta"].shift(lag)
        df[f"open_interest_lag_{lag}"] = df["open_interest"].shift(lag)

    # Bullish Buildup Conditions
    df['bullish_price'] = (df['close_lag_1'] > df['close_lag_2']) & \
                          (df['close_lag_2'] > df['close_lag_3']) & \
                          (df['close_lag_3'] > df['close_lag_4'])

    df['bullish_volume'] = (df['volume_lag_1'] > df['volume_lag_2']) & \
                           (df['volume_lag_2'] > df['volume_lag_3']) & \
                           (df['volume_lag_3'] > df['volume_lag_4'])

    df['bullish_buildup'] = (df['bullish_price'] & df['bullish_volume']).astype(int)

    # Bearish Buildup Conditions
    df['bearish_price'] = (df['close_lag_1'] < df['close_lag_2']) & \
                          (df['close_lag_2'] < df['close_lag_3']) & \
                          (df['close_lag_3'] < df['close_lag_4'])

    df['bearish_volume'] = (df['volume_lag_1'] > df['volume_lag_2']) & \
                           (df['volume_lag_2'] > df['volume_lag_3']) & \
                           (df['volume_lag_3'] > df['volume_lag_4'])

    df['bearish_buildup'] = (df['bearish_price'] & df['bearish_volume']).astype(int)

    # ✅ Price-Volume Momentum Features
    for lag in [1, 2, 3]:
        df[f'price_up_{lag}'] = (df[f'close_lag_{lag}'] > df[f'close_lag_{lag + 1}']).astype(int)
        df[f'volume_up_{lag}'] = (df[f'volume_lag_{lag}'] > df[f'volume_lag_{lag + 1}']).astype(int)

    # ✅ OBV Direction
    df["obv_direction_1"] = np.where(df["close_lag_1"] > df["close_lag_2"], df["volume_lag_1"], -df["volume_lag_1"])
    df["obv_direction_2"] = np.where(df["close_lag_2"] > df["close_lag_3"], df["volume_lag_2"], -df["volume_lag_2"])
    df["obv_direction_3"] = np.where(df["close_lag_3"] > df["close_lag_4"], df["volume_lag_3"], -df["volume_lag_3"])

    # ✅ Final Features
    df["bullish_score"] = df[['price_up_1', 'price_up_2', 'price_up_3']].sum(axis=1) + \
                          df[['volume_up_1', 'volume_up_2', 'volume_up_3']].sum(axis=1)
    df["cumulative_obv"] = df[['obv_direction_1', 'obv_direction_2', 'obv_direction_3']].sum(axis=1)

    # Confirm with RSI trend
    df['rsi_trend_up'] = (df['rsi_lag_1'] > df['rsi_lag_2']) & \
                         (df['rsi_lag_2'] > df['rsi_lag_3']) & \
                         (df['rsi_lag_3'] > df['rsi_lag_4'])

    df['bullish_confirmed'] = (df['bullish_buildup'] == 1) & (df['rsi_trend_up'] == 1)

    df['vol_q25_so_far'] = date_grp['volatility'].transform(
        lambda x: x.expanding().quantile(0.25)
    )
    df['low_vol_bullish'] = (df['bullish_buildup'] == 1) & (df['volatility'] < df['vol_q25_so_far'])

    # Recent periods matter more
    weights = [0.5, 0.3, 0.2]  # For lags 1, 2, 3
    df['weighted_bullish_score'] = (
            df['price_up_1'] * weights[0] + df['price_up_2'] * weights[1] + df['price_up_3'] * weights[2]
    )

    df['cumulative_day_volume'] = date_grp['volume'].transform(
        lambda x: x.expanding().sum()
    )
    df['day_high_so_far'] = date_grp['high'].transform(lambda x: x.expanding().max())
    df['day_low_so_far'] = date_grp['low'].transform(lambda x: x.expanding().min())
    df['distance_from_lb'] = (df['lower_band'] - df['close']) / df['close']
    df['distance_to_hb'] = (df['upper_band'] - df['close']) / df['close']
    df['distance_from_high'] = (df['day_high_so_far'] - df['close']) / df['close']
    df['distance_from_low'] = (df['close'] - df['day_low_so_far']) / df['close']
    df['volume_mean'] = date_grp['volume'].transform(
        lambda x: x.rolling(20, min_periods=1).mean())
    df['volume_std'] = date_grp['volume'].transform(
        lambda x: x.rolling(20, min_periods=1).std())
    df['volume_zscore'] = (df['volume'] - df['volume_mean']) / df['volume_std']
    df['intraday_trend_strength'] = (df['close'] - df['open']) / (
            df['day_high_so_far'] - df['day_low_so_far'] + 1e-6)

    df['cumulative_delta'] = date_grp['delta'].transform(lambda x: x.expanding().sum())

    df["implied_volatility"] = df.apply(calculate_iv, axis=1)
    df["implied_volatility"].fillna(df["implied_volatility"].median(), inplace=True)

    df['vix_ma'] = df['vix_close'].transform(
        lambda x: safe_rolling(x)
    )
    df['iv_skew'] = df['vix_close'] - df['vix_ma']

    df['volatility_adjusted_volume'] = df['volume'] * df['vix_ma']

    df['price_impact'] = (df['high'] - df['low']) / (
            df['volume'].replace(0, 1e-6) + 1e-6)

    df['gamma_exposure'] = df['gamma'] * df['volume'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    df['session_progress'] = date_grp.cumcount() / 375

    df['momentum_1'] = df['close'].transform(
        lambda x: x.pct_change(3)
    )
    df['momentum_2'] = df['close'].transform(
        lambda x: x.pct_change(5)
    )
    df['momentum_3'] = df['close'].transform(
        lambda x: x.pct_change(15)
    )
    df["close_per_1"] = df["close"] / df["close_lag_1"].replace(0, np.nan)
    df["close_per_2"] = df["close"] / df["close_lag_2"].replace(0, np.nan)
    df["close_per_3"] = df["close"] / df["close_lag_3"].replace(0, np.nan)
    df["close_per_4"] = df["close"] / df["close_lag_4"].replace(0, np.nan)
    df["close_per_5"] = df["close"] / df["close_lag_5"].replace(0, np.nan)

    df['normalized_close_5'] = (df['close'] - df['close'].rolling(5).mean()) / df['close'].rolling(
        5).std()
    df['normalized_close_15'] = (df['close'] - df['close'].rolling(15).mean()) / df['close'].rolling(
        15).std()
    df['normalized_close_30'] = (df['close'] - df['close'].rolling(30).mean()) / df['close'].rolling(
        30).std()

    df['sin_time'] = np.sin(2 * np.pi * df['session_progress'])
    df['cos_time'] = np.cos(2 * np.pi * df['session_progress'])

    df['open_interest_5'] = (df['open_interest'] - df['open_interest'].rolling(5).mean()) / df[
        'open_interest'].rolling(
        5).std()
    df['open_interest_15'] = (df['open_interest'] - df['open_interest'].rolling(15).mean()) / df[
        'open_interest'].rolling(15).std()
    df['open_interest_30'] = (df['open_interest'] - df['open_interest'].rolling(30).mean()) / df[
        'open_interest'].rolling(30).std()

    df['price_momentum'] = df['close_per_1'] * df['rsi']
    df['volatility_ratio'] = df['atr'] / df['bollinger_bandwidth']
    df['price_range'] = df['distance_from_high'] + df['distance_from_low']

    df["stoch_rsi"] = ta.momentum.stochrsi(df["close"], window=14)
    df["price_to_vwap"] = df["close"] / df["vwap"]
    df["volumexvwap"] = df["volume"] * df["vwap"]
    df["dist_to_fib_61_8"] = (df["close"] - df["fib_61_8"]) / df["close"]
    df["dist_to_fib_38_2"] = (df["close"] - df["fib_38_2"]) / df["close"]

    df['ewma_close_5'] = df['close'].ewm(span=5).mean()
    df['ewma_rsi_14'] = df['rsi'].ewm(span=14).mean()
    df['roc_5'] = df['close'].pct_change(5)
    df['momentum_acceleration'] = df['momentum_1'] - df['momentum_2']

    df['last_bullish'] = date_grp['bullish_buildup'].cumsum()
    df['time_since_bullish'] = df.groupby(['date_only', 'last_bullish']).cumcount()
    df['volatility_spike'] = (df['volatility'] > df['volatility'].rolling(20).mean() * 1.5).astype(int)

    df['close_to_20_high'] = df['close'] / df['close'].rolling(20).max()
    df['close_to_20_low'] = df['close'] / df['close'].rolling(20).min()

    df['rolling_return_5'] = df['close'].pct_change(periods=5).rolling(window=5).mean()
    df['rolling_return_15'] = df['close'].pct_change(periods=15).rolling(window=15).mean()

    df['oi_change'] = df['open_interest'].pct_change()
    df['oi_to_volume_ratio'] = df['oi_change'] / df['volume'].replace(0, 1)

    df['volume_surge'] = df['volume'] / df['volume'].rolling(10).mean()
    df['price_expansion'] = df['bollinger_bandwidth'] * df['rsi']

    df['vwap_zscore'] = (df['close'] - df['vwap']) / df['close'].rolling(20).std()

    df['deviation_from_high'] = df['close'] - df['close'].rolling(20).max()
    df['deviation_from_low'] = df['close'] - df['close'].rolling(20).min()

    returns = df['close'].pct_change()
    df['price_spike'] = (
            df['close'].pct_change() > returns.rolling(20).mean() + 2 * returns.rolling(
        20).std()).astype(
        int)

    df['macd_slope'] = df['macd'].diff()
    df['rsi_macd_divergence'] = df['rsi'] - df['macd']
    df['price_to_atr'] = df['close'] / df['atr']

    # Extract one row per day (no leakage)
    daily_summary = date_grp.agg({
        'underlying_open': 'first',
        'underlying_close': 'last',
        'underlying_high': 'max',
        'underlying_low': 'min',
        'vix_close': 'last'
    }).rename(columns={
        'underlying_open': 'daily_underlying_open',
        'underlying_close': 'daily_underlying_close',
        'underlying_high': 'daily_underlying_high',
        'underlying_low': 'daily_underlying_low',
        'vix_close': 'daily_vix_close'
    }).sort_index()

    # Shift to get previous day's values
    daily_summary['prev_day_close'] = daily_summary['daily_underlying_close'].shift(1)
    daily_summary['prev_day_open'] = daily_summary['daily_underlying_open'].shift(1)
    daily_summary['prev_day_high'] = daily_summary['daily_underlying_high'].shift(1)
    daily_summary['prev_day_low'] = daily_summary['daily_underlying_low'].shift(1)
    daily_summary['prev_day_vix_close'] = daily_summary['daily_vix_close'].shift(1)

    # Merge back
    df = df.merge(daily_summary, left_on='date_only', right_index=True, how='left')

    df.drop(columns=['daily_underlying_close', 'daily_vix_close', 'daily_underlying_high',
                     'daily_underlying_low'], inplace=True)

    # Calculate gap and change from previous day
    df['gap_pct'] = (df['daily_underlying_open'] - df['prev_day_close']) / df['prev_day_close']
    df['underlying_pct_from_prev_close'] = (df['underlying_close'] - df['prev_day_close']) / df['prev_day_close']
    df['vix_change_from_prev_day'] = (df['vix_close'] - df['prev_day_vix_close']) / df['prev_day_vix_close']

    df['gap_up'] = (df['gap_pct'] > 0.005).astype(int)
    df['gap_down'] = (df['gap_pct'] < -0.005).astype(int)

    df['vix_spike'] = (df['vix_change_from_prev_day'] > 0.03).astype(int)  # spike > 3%
    df['bias_pe_vix'] = df['vix_spike']

    df['combined_pe_bias'] = df['gap_down'] + df['bias_pe_vix']
    df['combined_ce_bias'] = df['gap_up'] * (1 - df['bias_pe_vix'])  # override CE bias if VIX is spiking

    df["premium_ratio_3min"] = df["close"] / df["close"].shift(3)
    df["premium_ratio_5min"] = df["close"] / df["close"].shift(5)
    df["premium_ratio_15min"] = df["close"] / df["close"].shift(15)

    df["underlying_pct_3min"] = df["underlying_close"].pct_change(3)
    df["underlying_pct_5min"] = df["underlying_close"].pct_change(5)
    df["underlying_pct_15min"] = df["underlying_close"].pct_change(15)

    df["underlying_move_vs_atr"] = (df["underlying_close"] - df["underlying_open"]) / df["underlying_atr"]

    df["underlying_hl_range"] = df["underlying_high"] - df["underlying_low"]
    df["underlying_range_ratio"] = df["underlying_hl_range"] / df["underlying_atr"].replace(0, 1e-6)

    df["underlying_break_high_20"] = (df["underlying_close"] > df["underlying_high"].rolling(20).max()).astype(
        int)
    df["underlying_break_low_20"] = (df["underlying_close"] < df["underlying_low"].rolling(20).min()).astype(
        int)

    df["premium_volatility_spike"] = (df["atr"] > df["atr"].rolling(20).mean() * 2).astype(int)

    d1 = (np.log(df["underlying_close"] / df["strike"]) + (0.06 + 0.5 * df["implied_volatility"] ** 2) * df[
        "days_to_expiry"] / 365) / (df["implied_volatility"] * np.sqrt(df["days_to_expiry"] / 365))
    df["corrected_delta"] = np.where(df["option_type"] == "CE", norm.cdf(d1), norm.cdf(d1) - 1)

    df["gamma_explosion"] = df["gamma"] * df["underlying_pct_5min"]

    df["market_open_volatility"] = ((df["hour"] == 9) & (df["date"].dt.minute <= 45)).astype(int)
    df["afternoon_volatility"] = ((df["hour"] == 14) & (df["date"].dt.minute >= 30)).astype(int)

    df["double_premium"] = (df["premium_ratio_5min"] >= 2).astype(int)
    df["triple_premium"] = (df["premium_ratio_15min"] >= 3).astype(int)

    df['premium_spike_1min'] = (df['close'] / df['close'].shift(1) > 1.25).astype(int)  # 25% jump in 1 min
    df['premium_spike_3min'] = (df['close'] / df['close'].shift(3) > 1.5).astype(int)  # 50% jump in 3 mins
    df['premium_spike_5min'] = (df['close'] / df['close'].shift(5) > 2).astype(int)  # 100% jump in 5 mins

    df['underlying_up_0_5pct'] = (df['underlying_close'].pct_change(periods=5) > 0.005).astype(int)
    df['underlying_up_1pct'] = (df['underlying_close'].pct_change(periods=5) > 0.01).astype(int)
    df['underlying_up_1_5pct'] = (df['underlying_close'].pct_change(periods=15) > 0.015).astype(int)

    df["premium_underlying_elasticity"] = (
            df["premium_ratio_5min"] / df["underlying_pct_5min"].replace(0, np.nan)).fillna(0)

    df["premium_volume_spike"] = (
            (df["premium_spike_3min"] == 1) & (df["volume"] > df["volume"].rolling(10).mean() * 2)).astype(int)

    df["atm_strike_distance_pct"] = (df["underlying_close"] - df["strike"]).abs() / df["underlying_close"]
    df["atm_proximity"] = (df["atm_strike_distance_pct"] < 0.005).astype(int)  # <0.5% from ATM

    df["delta_change_3min"] = df["corrected_delta"].diff(3)
    df["gamma_change_3min"] = df["gamma"].diff(3)
    df["rapid_gamma_spike"] = (df["gamma_change_3min"] > df["gamma"].rolling(15).std() * 2).astype(int)

    df["underlying_hammer"] = hammer(df["underlying_open"], df["underlying_high"],
                                     df["underlying_low"], df["underlying_close"])

    bullish, bearish = detect_engulfing(
        df["underlying_open"],
        df["underlying_close"]
    )

    df["underlying_bullish_engulfing"] = bullish
    df["underlying_bearish_engulfing"] = bearish

    df["premium_intrinsic_ratio"] = df["close"] / (df["intrinsic_value"].replace(0, np.nan))

    df["underlying_rapid_drop"] = (df["underlying_pct_3min"] < -0.0075).astype(int)
    df["vix_rapid_jump"] = (df["vix_change"] > 0.03).astype(int)

    df["high_profit_trade"] = ((df["premium_ratio_15min"] >= 2) | (df["premium_ratio_5min"] >= 1.5)).astype(int)

    df["premium_volatility_rank"] = df["atr"].rolling(100).rank(pct=True)

    df['underlying_pct_from_open'] = (df['underlying_close'] - df['underlying_open']) / df['underlying_open']
    df['underlying_pct_from_prev_close'] = (df['underlying_close'] - df['prev_day_close']) / df['prev_day_close']
    df['underlying_distance_points_from_open'] = df['underlying_close'] - df['underlying_open']
    df['underlying_distance_points_from_prev_close'] = df['underlying_close'] - df['prev_day_close']
    df['underlying_high_since_open'] = date_grp['underlying_high'].transform(
        lambda x: x.expanding().max())
    df['underlying_low_since_open'] = date_grp['underlying_low'].transform(
        lambda x: x.expanding().min())

    df['distance_from_open_high'] = (df['underlying_high_since_open'] - df['underlying_open']) / df['underlying_open']
    df['distance_from_open_low'] = (df['underlying_open'] - df['underlying_low_since_open']) / df['underlying_open']

    # Direction of underlying
    df['underlying_return_3min'] = df['underlying_close'].pct_change(3)
    df['underlying_return_5min'] = df['underlying_close'].pct_change(5)

    # Encode option type as a numeric feature
    df['is_ce'] = (df['option_type'] == 'CE').astype(int)
    df['is_pe'] = (df['option_type'] == 'PE').astype(int)

    # Directional alignment feature
    df['directional_alignment'] = np.where(
        (df['is_ce'] == 1) & (df['underlying_return_3min'] > 0), 1,
        np.where((df['is_pe'] == 1) & (df['underlying_return_3min'] < 0), 1, 0)
    )

    df['regime_flag'] = (df['date'] >= '2024-10-15').astype(int)
    df['sample_weight'] = np.where(df['regime_flag'] == 1, 1.25, 1.0)

    df['momentum_x_volatility'] = df['momentum_1'] * df['volatility']
    df['vwap_vs_rsi'] = df['price_to_vwap'] * df['rsi']
    df['delta_x_gamma'] = df['delta'] * df['gamma']

    # ✅ 1. Candlestick Body and Wick Strength
    df['candle_body_pct'] = ((df['close'] - df['open']) / df['open']) * 100
    df['candle_range_pct'] = ((df['high'] - df['low']) / df['low']) * 100

    # ✅ 2. Upper and Lower Wicks
    df['upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
    df['lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']

    # ✅ 3. Body-to-Range Ratio (candlestick structure indicator)
    df['body_to_range_ratio'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-6)

    # ✅ 4. Volume Relative to Last N candles (signal strength)
    df['volume_vs_5'] = df['volume'] / df['volume'].rolling(5).mean()
    df['volume_vs_10'] = df['volume'] / df['volume'].rolling(10).mean()

    # ✅ 5. Strong Bullish/Bearish Candles with Volume Confirmation
    df['strong_bullish_candle'] = (
            (df['candle_body_pct'] > 2) &  # >2% body gain
            (df['close'] > df['open']) &
            (df['volume_vs_5'] > 1.5)
    ).astype(int)

    df['strong_bearish_candle'] = (
            (df['candle_body_pct'] < -2) &
            (df['close'] < df['open']) &
            (df['volume_vs_5'] > 1.5)
    ).astype(int)

    # ✅ 6. Rejection Candles (small body + large wick + high volume)
    df['rejection_candle'] = (
            (df['body_to_range_ratio'] < 0.25) &
            ((df['upper_wick'] > 0.5) | (df['lower_wick'] > 0.5)) &
            (df['volume_vs_5'] > 1.5)
    ).astype(int)

    candlestick_features = [
        'candle_body_pct', 'candle_range_pct', 'upper_wick', 'lower_wick',
        'body_to_range_ratio', 'volume_vs_5', 'volume_vs_10',
        'strong_bullish_candle', 'strong_bearish_candle', 'rejection_candle'
    ]

    for col in candlestick_features:
        df[f'{col}_lag_1'] = df[col].shift(1)
        df[f'{col}_lag_2'] = df[col].shift(2)

    # Bias based on PCR + direction
    df['bias_ce'] = ((df['pcr'] > 1.2) & (df['underlying_pct_from_prev_close'] > 0)).astype(int)
    df['bias_pe'] = ((df['pcr'] < 0.8) & (df['underlying_pct_from_prev_close'] < 0)).astype(int)

    # OI dominance levels
    for level in ['1', '2', '3']:
        df[f'support_dominant_{level}'] = (df[f'support_oi_{level}'] > df[f'resistance_oi_{level}']).astype(int)
        df[f'resistance_dominant_{level}'] = (df[f'resistance_oi_{level}'] > df[f'support_oi_{level}']).astype(int)
        df[f'support_score_{level}'] = df[f'support_dominant_{level}'] * df[f'support_oi_{level}'] / (
                df[f'support_distance_{level}'] + 1e-6)
        df[f'resistance_score_{level}'] = df[f'resistance_dominant_{level}'] * df[f'resistance_oi_{level}'] / (
                df[f'resistance_distance_{level}'] + 1e-6)

    df['support_strength'] = df[[f'support_score_{i}' for i in [1, 2, 3]]].sum(axis=1)
    df['resistance_strength'] = df[[f'resistance_score_{i}' for i in [1, 2, 3]]].sum(axis=1)

    # Bias based on strength comparison
    df['oi_bias_ce'] = (df['support_strength'] > df['resistance_strength']).astype(int)
    df['oi_bias_pe'] = (df['support_strength'] < df['resistance_strength']).astype(int)

    # Distance proximity
    df['near_support_1'] = (df['support_distance_1'] < 20).astype(int)
    df['near_resistance_1'] = (df['resistance_distance_1'] < 20).astype(int)

    df['support_hold_ce'] = ((df['support_strength'] > df['resistance_strength']) & (df['near_support_1'] == 1)).astype(
        int)
    df['resistance_block_pe'] = (
            (df['resistance_strength'] > df['support_strength']) & (df['near_resistance_1'] == 1)).astype(int)

    # Final signal — same logic for both CE and PE
    df['final_ce_signal'] = (df['bias_ce'] + df['oi_bias_ce'] + df['support_hold_ce']) >= 2
    df['vix_spike_pe'] = (df['vix_change_from_prev_day'] > .25).astype(int)
    df['final_pe_signal'] = (df['bias_pe'] + df['oi_bias_pe'] + df['resistance_block_pe'] + df['vix_spike_pe']) >= 2

    df['rr_q25_so_far'] = date_grp['rolling_range'].transform(
        lambda x: x.expanding().quantile(0.25)
    )
    df['consolidation_breakout'] = ((df['close'] > df['high'].rolling(20).max()) &
                                    (df['rolling_range'] < df['rr_q25_so_far'])).astype(int)

    # ✅ Custom Signal: MACD crossover
    df['macd_signal_crossover'] = ((df['macd'] > df['macd_signal']) &
                                   (df['macd'].shift(1) < df['macd_signal'].shift(1))).astype(int)

    # ✅ RSI zones
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)

    df['signal_buy'] = ((df['final_ce_signal'] == 1) & (df['macd_signal_crossover'] == 1) &
                        (df['support_hold_ce'] == 1)).astype(int)

    # ✅ Strong support within ideal range
    df['strong_support_zone'] = (
            (df['support_distance_1'] <= 60) &
            (df['support_strength'] >= 0.2)
    ).astype(int)

    # ✅ Strong resistance within ideal range
    df['strong_resistance_zone'] = (
            (df['resistance_distance_1'] <= 60) &
            (df['resistance_strength'] >= 0.2)
    ).astype(int)

    # ✅ Weak support (far + low strength)
    df['weak_support_zone'] = (
            (df['support_distance_1'] > 80) &
            (df['support_strength'] < 0.15)
    ).astype(int)

    # ✅ Weak resistance (far + low strength)
    df['weak_resistance_zone'] = (
            (df['resistance_distance_1'] > 80) &
            (df['resistance_strength'] < 0.15)
    ).astype(int)

    # ✅ OI Skew Signal: Support more dominant than resistance by large margin
    df['oi_support_dominant'] = (
            df['support_strength'] > df['resistance_strength'] * 1.5
    ).astype(int)

    df['oi_resistance_dominant'] = (
            df['resistance_strength'] > df['support_strength'] * 1.5
    ).astype(int)

    # ✅ Risk zone if both S/R are close (squeeze breakout potential)
    df['squeeze_zone'] = (
            (df['support_distance_1'] <= 50) &
            (df['resistance_distance_1'] <= 50)
    ).astype(int)

    df['support_strength_x_distance'] = df['support_strength'] / (df['support_distance_1'] + 1e-6)
    df['resistance_strength_x_distance'] = df['resistance_strength'] / (df['resistance_distance_1'] + 1e-6)

    df['delta_x_oi'] = df['delta'] * df['open_interest']
    df['vix_x_iv'] = df['vix_close'] * df['implied_volatility']
    df['close_x_gap'] = df['close'] * df['gap_pct']

    df['support_x_t'] = np.where(
        df['support_strength_1'] != 0,
        df['support_distance_1'] / df['support_strength_1'],
        np.nan
    )

    df['resistance_x_t'] = np.where(
        df['resistance_strength_1'] != 0,
        df['resistance_distance_1'] / df['resistance_strength_1'],
        np.nan
    )

    df = df.set_index('date')

    # Create a bucket label: every 15 minutes starting 9:15
    df['bucket_start'] = ((df.index - pd.Timestamp(df.index[0].date()) - pd.Timedelta(minutes=15))
                          .floor('15min') + pd.Timestamp(df.index[0].date()) + pd.Timedelta(minutes=15))

    # Fix for pre-9:15 data
    df.loc[df.index < df['bucket_start'].min(), 'bucket_start'] = pd.Timestamp(df.index[0].date()) + pd.Timedelta(
        minutes=9 * 60 + 15)

    # Group by bucket and within bucket calculate live open, high, low, close till now
    df['underlying_15_open'] = df.groupby('bucket_start')['underlying_open'].transform('first')
    df['underlying_15_high'] = df.groupby('bucket_start')['underlying_high'].expanding().max().reset_index(level=0,
                                                                                                           drop=True)
    df['underlying_15_low'] = df.groupby('bucket_start')['underlying_low'].expanding().min().reset_index(level=0,
                                                                                                         drop=True)
    df['underlying_15_close'] = df['underlying_close']

    df = df.reset_index()

    # Now directly add indicators using 15min close
    df["underlying_15_sma_20"] = ta.trend.sma_indicator(df["underlying_15_close"], window=20)
    df["underlying_15_sma_50"] = ta.trend.sma_indicator(df["underlying_15_close"], window=50)
    df["underlying_15_sma_200"] = ta.trend.sma_indicator(df["underlying_15_close"], window=200)
    df["underlying_15_rsi"] = ta.momentum.rsi(df["underlying_15_close"], window=14)

    df["underlying_15_sma_crossover"] = (df["underlying_15_sma_50"] > df["underlying_15_sma_200"]).astype(int)
    df["underlying_15_trend_up"] = df["underlying_15_sma_20"].diff() > 0
    df["underlying_15_rsi_trend"] = df["underlying_15_rsi"].diff() > 0

    df["underlying_15_range"] = df["underlying_15_high"] - df["underlying_15_low"]
    df["underlying_15_range_rank"] = df["underlying_15_range"].rolling(20).rank(pct=True)

    df['htf_breakout_confirmed'] = (
            (df['underlying_15_sma_crossover'] == 1) &
            (df['underlying_15_rsi_trend'] == 1) &
            (df['underlying_15_trend_up'] == 1) &
            (df['underlying_15_range_rank'] > 0.8)
    ).astype(int)

    df['htf_breakdown_confirmed'] = (
            (df['underlying_15_sma_crossover'] == 0) &
            (df['underlying_15_rsi_trend'] == 0) &
            (df['underlying_15_trend_up'] == 0) &
            (df['underlying_15_range_rank'] > 0.8)
    ).astype(int)

    df['htf_trend_confirmation'] = np.select(
        [(df['htf_breakout_confirmed'] == 1),
         (df['htf_breakdown_confirmed'] == 1)],
        [1, -1],
        default=0
    )

    # ✅ VWAP Slope - Detect early trend shift near vwap
    df['vwap_slope_3'] = df['vwap'].diff(1).rolling(3).mean()
    df['vwap_slope_direction'] = (df['vwap_slope_3'] > 0).astype(int)

    # ✅ Price Compression → Likely breakout
    df['compression_score'] = (df['high'] - df['low']) / (df['atr'] + 1e-6)
    df['compression_detected'] = (df['compression_score'] < 0.6).astype(int)  # adjustable

    # ✅ Price CCI — Leading momentum shift detector
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=14)
    df['cci_breakout'] = ((df['cci'] > 100) & (df['cci'].shift(1) <= 100)).astype(int)
    df['cci_breakdown'] = ((df['cci'] < -100) & (df['cci'].shift(1) >= -100)).astype(int)

    df['green_streak_len'] = date_grp['green_streak'].cumsum() * df['green_streak']
    df['red_streak_len'] = date_grp['red_streak'].cumsum() * df['red_streak']

    # ✅ Mean Reversion Alert
    df['distance_to_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    df['mean_reversion_signal'] = ((df['distance_to_vwap'].abs() > 0.01) &  # >1%
                                   (df['vwap_slope_3'] * df['distance_to_vwap'] < 0)).astype(int)

    # ✅ Time of Day Bias (pre-breakout windows)
    df['time_bin'] = df['date'].dt.hour * 60 + df['date'].dt.minute
    df['pre_breakout_time'] = ((df['time_bin'] >= 930) & (df['time_bin'] <= 1045)).astype(int)

    # ✅ Price Thrust (large directional bar with volume)
    df['price_thrust'] = ((df['candle_body_pct'].abs() > 1.5) & (df['volume_vs_5'] > 1.5)).astype(int)

    # ✅ Micro Pullback (2-bar dip but still bullish)
    df['micro_pullback'] = (
            (df['close_lag_1'] < df['close_lag_2']) &
            (df['close'] > df['close_lag_1']) &
            (df['rsi'] > 50)
    ).astype(int)

    # ✅ Breakout Base Structure (range + volume squeeze)
    df['breakout_base'] = (
            (df['bollinger_bandwidth'] < df['bollinger_bandwidth'].rolling(20).mean()) &
            (df['volume_zscore'] < 0)
    ).astype(int)

    df['leading_score'] = (
                                  df['breakout_base'] +
                                  df['cci_breakout'] +
                                  df['vwap_slope_direction'] +
                                  df['price_thrust'] +
                                  df['green_streak_len'].apply(lambda x: 1 if x >= 3 else 0)
                          ) / 5.0  # scale between 0 to 1
    # Option vs underlying relative momentum
    df['rs_ratio'] = df['close'].pct_change(5) / df['underlying_close'].pct_change(5)
    df['rs_ratio_signal'] = (df['rs_ratio'] > 1.2).astype(int)  # Option outperforming underlying
    # Spike in premium but no IV spike → real demand
    df['iv_decline'] = df['implied_volatility'].diff()
    df['premium_spike_but_iv_down'] = ((df['premium_spike_3min'] == 1) & (df['iv_decline'] < 0)).astype(int)
    # Heikin-Ashi close formula
    df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['ha_trend_strength'] = df['ha_close'].diff().rolling(3).mean()
    # Pre-breakout squeeze + trend confirmation
    df['trend_and_squeeze'] = (
            (df['ema_crossover'] == 1) &
            (df['bollinger_bandwidth'] < df['bollinger_bandwidth'].rolling(20).mean()) &
            (df['rsi'] > 50)
    ).astype(int)
    # VIX surging but price not falling
    df['vix_divergence'] = ((df['vix_change'] > 0.03) & (df['underlying_return_3min'] > 0)).astype(int)

    # Simple 3-bar gap detection
    df['fvg_up'] = (df['low'].shift(1) > df['high'].shift(2)).astype(int)
    df['fvg_down'] = (df['high'].shift(1) < df['low'].shift(2)).astype(int)
    df['smart_money_footprint'] = (
            df['premium_spike_but_iv_down'] +
            df['fvg_up'] +
            df['trend_and_squeeze'] +
            df['vix_divergence']
    ).clip(0, 1)  # Binary signal

    df['hidden_bullish'] = (
            (df['close'] < df['open']) &  # red candle
            (df['close'] > df['close_lag_1']) &  # but closing higher than previous
            (df['volume'] > df['volume_lag_1'])
    ).astype(int)

    df['hidden_bearish'] = (
            (df['close'] > df['open']) &
            (df['close'] < df['close_lag_1']) &
            (df['volume'] > df['volume_lag_1'])
    ).astype(int)

    df['volume_shock_reversal'] = (
            (df['volume'] > df['volume'].rolling(20).mean() * 2) &
            (df['rejection_candle'] == 1)
    ).astype(int)

    df['close_below_half'] = (df['close'] < (df['high'] + df['low']) / 2).astype(int)
    df['rejection_upper_half'] = (
            (df['close_lag_1'] > (df['high'] + df['low']) / 2) &
            (df['close_below_half'] == 1)
    ).astype(int)
    df['bull_strength'] = df['upper_wick'].rolling(5).mean()
    df['bear_strength'] = df['lower_wick'].rolling(5).mean()
    df['wick_imbalance'] = df['bull_strength'] - df['bear_strength']
    df['elasticity_ratio'] = df['premium_ratio_3min'] / (df['underlying_pct_3min'].replace(0, np.nan))
    df['elasticity_failure'] = (df['elasticity_ratio'] < 1).astype(int)
    df['squeeze_exhaustion'] = (
            (df['bollinger_bandwidth'] < df['bollinger_bandwidth'].rolling(20).mean()) &
            (df['rsi'] > 60) &
            (df['volume'] > df['volume'].rolling(20).mean() * 1.5)
    ).astype(int)
    df['lower_high'] = (df['high'] < df['high'].shift(1)) & (df['high'].shift(1) < df['high'].shift(2))
    df['higher_low'] = (df['low'] > df['low'].shift(1)) & (df['low'].shift(1) > df['low'].shift(2))
    df['smart_price_action_trigger'] = (
            df['hidden_bullish'] +
            df['rejection_upper_half'] +
            df['squeeze_exhaustion'] +
            df['lower_high'].astype(int)
    ).clip(0, 1)

    df['daily_high_so_far'] = date_grp['high'].cummax()
    df['daily_low_so_far'] = date_grp['low'].cummin()
    df['daily_range'] = df['daily_high_so_far'] - df['daily_low_so_far']
    df['underlying_day_high_so_far'] = date_grp['underlying_high'].cummax()
    df['underlying_day_low_so_far'] = date_grp['underlying_low'].cummin()
    df['underlying_daily_range'] = df['underlying_day_high_so_far'] - df['underlying_day_low_so_far']
    df['underlying_distance_from_high'] = (df['underlying_day_high_so_far'] - df['underlying_close']) / df[
        'underlying_close']
    df['underlying_distance_from_low'] = (df['underlying_close'] - df['underlying_day_low_so_far']) / df[
        'underlying_close']
    df['atr_vs_range_ratio'] = df['atr'] / (df['daily_range'] + 1e-6)
    df['range_squeeze_alert'] = (df['atr_vs_range_ratio'] > 1.5).astype(int)
    df['iv_vs_vix_diff'] = df['implied_volatility'] - df['vix_close']
    df['option_undervalued'] = (df['iv_vs_vix_diff'] < -2).astype(int)
    df['option_overvalued'] = (df['iv_vs_vix_diff'] > 2).astype(int)
    df['micro_breakout'] = ((df['close'] > df['high'].shift(1)) & (df['volume'] > df['volume_lag_1'])).astype(int)
    df['micro_breakout_count'] = df['micro_breakout'].rolling(5).sum()
    df['expiry_rush'] = ((df['minutes_to_expiry'] < 60) & (df['premium_spike_5min'] == 1)).astype(int)
    df['vwap_upper'] = df['vwap'] + df['atr']
    df['vwap_lower'] = df['vwap'] - df['atr']
    df['snap_reversal_up'] = ((df['close'] < df['vwap_lower']) & (df['close_lag_1'] > df['vwap_lower'])).astype(int)
    df['snap_reversal_down'] = ((df['close'] > df['vwap_upper']) & (df['close_lag_1'] < df['vwap_upper'])).astype(int)
    df['underlying_moved'] = (df['underlying_pct_3min'].abs() > 0.002).astype(int)
    df['premium_did_not_move'] = (df['premium_ratio_3min'] < 1.05).astype(int)
    df['mispricing_window'] = (df['underlying_moved'] == 1) & (df['premium_did_not_move'] == 1)
    df['theta_crush_risk'] = ((df['is_ce'] == 1) & (df['rsi'] < 40) & (df['minutes_to_expiry'] < 45)).astype(int)
    df['explosive_entry_score'] = (
                                          df['micro_breakout_count'].clip(0, 2) +
                                          df['option_undervalued'] +
                                          df['expiry_rush'] +
                                          df['snap_reversal_up'] +
                                          df['range_squeeze_alert']
                                  ) / 5.0
    df['break_count'] = df['micro_breakout'].rolling(5).sum()
    df['consolidation_before_breakout'] = ((df['breakout_base'].shift(1) == 1) & (df['micro_breakout'] == 1)).astype(
        int)
    df['rsi_slope_5'] = df['rsi'].diff().rolling(5).mean()
    df['oi_ce_change'] = df[df['option_type'] == 'CE'].groupby('date_only')['open_interest'].transform(
        'sum').pct_change()
    df['smart_money_score'] = (
                                      df['premium_spike_3min'] + df['premium_spike_but_iv_down'] + df['volume_surge']
                              ) / 3.0
    df['failed_breakout'] = (
            (df['high'] > df['high'].shift(1)) & (df['close'] < df['close'].shift(1))
    ).astype(int)

    # 1. Moving averages & crossovers
    df["u_sma_20"] = ta.trend.sma_indicator(df["underlying_close"], window=20)
    df["u_sma_50"] = ta.trend.sma_indicator(df["underlying_close"], window=50)
    df["u_sma_200"] = ta.trend.sma_indicator(df["underlying_close"], window=200)
    df["u_ema_12"] = ta.trend.ema_indicator(df["underlying_close"], window=12)
    df["u_ema_26"] = ta.trend.ema_indicator(df["underlying_close"], window=26)
    df["u_ema_xover"] = (df["u_ema_12"] > df["u_ema_26"]).astype(int)

    # 2. MACD & TRIX
    macd = ta.trend.MACD(df["underlying_close"], window_slow=26, window_fast=12, window_sign=9)
    df["u_macd"] = macd.macd()
    df["u_macd_signal"] = macd.macd_signal()
    df["u_macd_hist"] = macd.macd_diff()
    df["u_trix"] = ta.trend.trix(df["underlying_close"], window=15)

    # 3. Momentum & oscillators
    df["u_rsi"] = ta.momentum.rsi(df["underlying_close"], window=14)
    df["u_rsi_slope"] = df["u_rsi"].diff()
    df["u_stochrsi"] = ta.momentum.stochrsi(df["underlying_close"], window=14)
    df["u_roc_5"] = df["underlying_close"].pct_change(5)
    df["u_cci"] = ta.trend.cci(df["underlying_high"], df["underlying_low"], df["underlying_close"], window=20)
    df["u_willr"] = ta.momentum.williams_r(df["underlying_high"], df["underlying_low"], df["underlying_close"], lbp=14)

    # 4. Volatility & channels
    df["u_donch_high_20"] = df["underlying_high"].rolling(20).max()
    df["u_donch_low_20"] = df["underlying_low"].rolling(20).min()
    # Keltner: typical price MA ± ATR
    tp = (df["underlying_high"] + df["underlying_low"] + df["underlying_close"]) / 3
    df["u_kelt_ma"] = tp.rolling(20).mean()
    df["u_kelt_up"] = df["u_kelt_ma"] + 1.5 * df["underlying_atr"]
    df["u_kelt_lo"] = df["u_kelt_ma"] - 1.5 * df["underlying_atr"]
    df["u_bb_bw"] = (ta.volatility.bollinger_hband(df["underlying_close"], window=20)
                     - ta.volatility.bollinger_lband(df["underlying_close"], window=20)) \
                    / ta.volatility.bollinger_mavg(df["underlying_close"], window=20)

    # 5. Volume & flow
    # df['u_obv'] = date_grp.apply(
    #     lambda g: ta.volume.on_balance_volume(g['underlying_close'], g['underlying_volume'])
    # ).reset_index(level=0, drop=True)

    raw_obv = ta.volume.on_balance_volume(df["underlying_close"],
                                          df["underlying_volume"])
    df["u_obv"] = raw_obv - raw_obv.groupby(df["date_only"]).transform("first")

    df["u_chaikin"] = ta.volume.chaikin_money_flow(df["underlying_high"], df["underlying_low"],
                                                   df["underlying_close"], df["underlying_volume"], window=20)

    # 6. Pivot points (classic)
    pivot = (df["underlying_high"] + df["underlying_low"] + df["underlying_close"]) / 3
    df["u_pivot"] = pivot
    df["u_r1"] = 2 * pivot - df["underlying_low"]
    df["u_s1"] = 2 * pivot - df["underlying_high"]
    df["u_r2"] = pivot + (df["underlying_high"] - df["underlying_low"])
    df["u_s2"] = pivot - (df["underlying_high"] - df["underlying_low"])

    # 7. Session & higher-TF
    df["u_open_range_hi"] = df.loc[df["hour"].between(9, 9) & (df["date"].dt.minute <= 30), "underlying_high"] \
        .expanding().max()
    df["u_open_range_lo"] = df.loc[df["hour"].between(9, 9) & (df["date"].dt.minute <= 30), "underlying_low"] \
        .expanding().min()
    # 5-min SMA on 1-min data to mimic higher TF
    df["u_5m_sma"] = df["underlying_close"].rolling(5).mean()
    df["u_15m_sma"] = df["underlying_close"].rolling(15).mean()

    # ────────────────────────────────────────────────────
    # ⚙️ 1m Underlying Volume & OI Features: VWAP, Pivot, Momentum
    # ────────────────────────────────────────────────────

    # 1. Typical price & session VWAP (reset each day)
    df["typ_price"] = (df["underlying_high"] + df["underlying_low"] + df["underlying_close"]) / 3

    # cumulative TP×Vol and Vol since midnight
    # cumulative typical‐price × volume, shifted one bar to avoid leakage
    tp_vol = df["typ_price"] * df["underlying_volume"]
    df["cum_tp_vol"] = tp_vol.groupby(df["date_only"]).cumsum()

    # cumulative volume, shifted
    df["cum_vol"] = df["underlying_volume"].groupby(df["date_only"]).cumsum()

    # rebuilt VWAP
    df["vwap_1m"] = (df["cum_tp_vol"] / df["cum_vol"]).fillna(method="ffill")
    df["dist_from_vwap"] = (df["underlying_close"] - df["vwap_1m"]) / df["vwap_1m"]

    # 2. Previous-day Pivot Points (Classic)
    #    compute daily OHLC
    daily = df.set_index("date").groupby(pd.Grouper(freq="D")).agg({
        "underlying_high": "max",
        "underlying_low": "min",
        "underlying_close": "last"
    })
    # pivot formulas
    daily["pivot"] = (daily["underlying_high"] + daily["underlying_low"] + daily["underlying_close"]) / 3
    daily["r1"] = 2 * daily["pivot"] - daily["underlying_low"]
    daily["s1"] = 2 * daily["pivot"] - daily["underlying_high"]
    daily["r2"] = daily["pivot"] + (daily["underlying_high"] - daily["underlying_low"])
    daily["s2"] = daily["pivot"] - (daily["underlying_high"] - daily["underlying_low"])
    # shift to get *previous* day values
    for col in ["pivot", "r1", "s1", "r2", "s2"]:
        daily[f"prev_{col}"] = daily[col].shift(1)
    # merge back on minute df
    df = df.merge(
        daily[[f"prev_{c}" for c in ["pivot", "r1", "s1", "r2", "s2"]]].rename(
            columns={f"prev_{c}": f"prev_day_{c}" for c in ["pivot", "r1", "s1", "r2", "s2"]}
        ),
        left_on=pd.to_datetime(df["date_only"]),
        right_index=True,
        how="left"
    )

    # 3. One-minute momentum / price-move features
    #    (all shifted so model only sees past)
    df["u_ret_1m"] = df["underlying_close"].pct_change()
    df["u_price_diff"] = (df["underlying_close"] - df["underlying_open"])

    # 4. Open Interest flow & OI-to-Volume
    df["oi_change_1m"] = df["open_interest"].diff()
    df["oi_to_vol"] = df["open_interest"] / (df["underlying_volume"].replace(0, 1))
    df["oi_vol_ratio_5"] = df["open_interest"].diff().rolling(5).mean() \
                           / (df["underlying_volume"].rolling(5).mean().replace(0, 1))

    # 5. Volume-based oscillators
    df["u_mfi"] = ta.volume.money_flow_index(
        df["underlying_high"], df["underlying_low"],
        df["underlying_close"], df["underlying_volume"], window=14
    )

    # 5-min Underlying Return
    df['u_return_5m'] = df['underlying_close'].pct_change(5)

    df['momentum_score'] = (
                                   df['u_ret_1m'].rolling(300).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]) +
                                   df['u_return_5m'].rolling(300).apply(
                                       lambda x: pd.Series(x).rank(pct=True).iloc[-1]) +
                                   (1 - df['dist_from_vwap'].abs().rolling(300).apply(
                                       lambda x: pd.Series(x).rank(pct=True).iloc[-1])) +
                                   df['oi_vol_ratio_5'].rolling(300).apply(
                                       lambda x: pd.Series(x).rank(pct=True).iloc[-1])
                           ) / 4.0

    # Rolling 5min OI Change (underlying)
    df['rolling_oi_delta_5m'] = df['underlying_oi'].diff(5)

    # Rolling 5min Volume/OI Ratio
    df['volume_oi_ratio_5m'] = (
            df['underlying_volume'].rolling(5).mean() /
            df['underlying_oi'].rolling(5).mean()
    )

    # 1-Hour (60min) SMA (Trend Confirmation)
    df['u_1h_sma_20'] = df['underlying_close'].rolling(60).mean()
    df['u_1h_trend_up'] = (df['u_1h_sma_20'].diff() > 0).astype(int)

    # 15min Bollinger Breakout
    df['underlying_15m_bb_breakout'] = (
            df['underlying_15_close'] > df['underlying_15_high'].rolling(20).max()
    ).astype(int)

    # VWAP Slope Reversal
    df['vwap_slope_5'] = df['vwap'].diff(5)
    df['vwap_slope_reversal'] = (
            (df['vwap_slope_5'] * df['vwap_slope_5']) < 0
    ).astype(int)

    # 1) Create a mask for rows between 09:15:00 (inclusive) and 09:30:00 (exclusive)
    mask = (
            (df['date'].dt.time >= time(9, 15)) &
            (df['date'].dt.time < time(9, 30))
    )

    # 2) Take only those rows and compute an expanding max/min within each date
    df_open = df[mask].copy()

    df_open['opening_range_high'] = (
        df_open
            .groupby('date_only')['underlying_high']
            .expanding()
            .max()
            .reset_index(level=0, drop=True)
    )

    df_open['opening_range_low'] = (
        df_open
            .groupby('date_only')['underlying_low']
            .expanding()
            .min()
            .reset_index(level=0, drop=True)
    )

    # 3) Merge those back onto the original df (on both date and date_only)
    df = df.merge(
        df_open[['date', 'date_only', 'opening_range_high', 'opening_range_low']],
        on=['date', 'date_only'],
        how='left'
    )

    # 4) After 09:30, you’ll still have NaN for those two columns—forward‐fill each day
    df[['opening_range_high', 'opening_range_low']] = (
        df
            .groupby('date_only')[['opening_range_high', 'opening_range_low']]
            .ffill()
    )

    # Apply ORB breakout logic only after 09:30
    df['time'] = df['date'].dt.time
    df['orb_breakout'] = ((df['time'] > pd.to_datetime('09:30').time()) &
                          (df['underlying_close'] > df['opening_range_high'])).astype(int)
    df['orb_breakdown'] = ((df['time'] > pd.to_datetime('09:30').time()) &
                           (df['underlying_close'] < df['opening_range_low'])).astype(int)
    df.drop(columns=['time'], inplace=True)

    # Morning, Midday, Afternoon Sessions
    df['minute_of_day'] = df['date'].dt.hour * 60 + df['date'].dt.minute
    df['session_morning'] = ((df['minute_of_day'] >= 555) & (df['minute_of_day'] < 690)).astype(int)
    df['session_midday'] = ((df['minute_of_day'] >= 690) & (df['minute_of_day'] < 810)).astype(int)
    df['session_afternoon'] = (df['minute_of_day'] >= 810).astype(int)

    # 5-min underlying momentum
    df['underlying_momentum_5m'] = df['underlying_close'].pct_change(5)

    # Volatility Compression Rank (lower = compression)
    rolling_std = df['underlying_close'].rolling(50).std()
    df['volatility_compression_rank'] = rolling_std.rolling(300).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

    # Underlying 5-min Range Expansion
    df['underlying_range_expansion_5m'] = (
            df['underlying_high'].rolling(5).max() - df['underlying_low'].rolling(5).min()
    )

    # Underlying Price Momentum x Range Expansion
    df['momentum_range_combo'] = (
            df['underlying_momentum_5m'] * df['underlying_range_expansion_5m']
    )

    # Distance from VWAP
    df['u_distance_to_vwap'] = (
            (df['underlying_close'] - df['underlying_vwap']) / (df['underlying_vwap'] + 1e-6)
    )

    # Near VWAP Zone (<0.5% proximity)
    df['u_near_vwap_zone'] = (df['u_distance_to_vwap'].abs() < 0.005).astype(int)
    df['u_vwap_slope_5'] = df['underlying_vwap'].diff(5)
    # Micro Mean Reversion Risk
    df['u_mean_reversion_signal'] = (
            (df['u_distance_to_vwap'].abs() > 0.01) &
            (df['u_vwap_slope_5'] * df['u_distance_to_vwap'] < 0)
    ).astype(int)

    # High ATR regime (high volatility environment)
    df['high_atr_regime'] = (
            df['atr'] > df['atr'].rolling(20).mean() * 1.5
    ).astype(int)

    # Low ATR regime
    df['low_atr_regime'] = (
            df['atr'] < df['atr'].rolling(20).mean() * 0.8
    ).astype(int)

    # Range Compression Breakout
    df['range_breakout'] = (
            (df['underlying_15_range'] < df['underlying_15_range'].rolling(20).mean()) &
            (df['underlying_15m_bb_breakout'] == 1)
    ).astype(int)

    df['break_prev_day_high'] = (df['underlying_close'] > df['prev_day_high']).astype(int)
    df['break_prev_day_low'] = (df['underlying_close'] < df['prev_day_low']).astype(int)

    # Gap Open Strength (Gap % / ATR)
    df['gap_open_strength'] = (df['gap_pct'].abs() / (df['atr'] + 1e-6))

    # ✅ Correct cumulative session return without index mismatch
    df['cumulative_return_session'] = date_grp['underlying_close'].transform(
        lambda x: x.pct_change().cumsum()
    )
    # Short-term pullback detector
    df['short_term_pullback'] = (
            (df['underlying_close'] < df['underlying_close'].rolling(5).mean()) &
            (df['underlying_15_sma_crossover'] == 1)
    ).astype(int)

    # Underlying Compression Breakout
    df['compression_breakout'] = (
            (df['underlying_15_range_rank'] < 0.3) &
            (df['underlying_close'] > df['underlying_15_high'].rolling(20).max())
    ).astype(int)

    # VWAP reclaim after loss
    df['vwap_reclaim'] = (
            (df['underlying_close'] < df['vwap']) &
            (df['underlying_close'] > df['vwap'])
    ).astype(int)

    # ATR Spike Detection
    df['atr_spike'] = (
            df['atr'] > df['atr'].rolling(20).mean() * 2
    ).astype(int)

    # Smart Money Flow Index (simple proxy)
    df['smart_money_flow'] = (
            (df['u_return_5m'] > 0) &
            (df['underlying_volume'] > df['underlying_volume'].rolling(5).mean())
    ).astype(int)

    # Underlying Window & Lookbacks
    df["underlying_return_30min"] = df["underlying_close"].pct_change(30)
    df["underlying_return_60min"] = df["underlying_close"].pct_change(60)
    df["underlying_volatility_30min"] = df["underlying_close"].rolling(30).std()
    df["underlying_volatility_60min"] = df["underlying_close"].rolling(60).std()
    df["underlying_volume_mean_30min"] = df["underlying_volume"].rolling(30).mean()
    df["underlying_oi_change_30min"] = df["underlying_oi"].diff(30)
    df["underlying_position_vs_30min_high"] = df["underlying_close"] / df["underlying_high"].rolling(30).max()
    df["underlying_position_vs_30min_low"] = df["underlying_close"] / df["underlying_low"].rolling(30).min()

    # Options Window & Lookbacks
    df["premium_return_30min"] = df["close"].pct_change(30)
    df["premium_return_60min"] = df["close"].pct_change(60)
    df["iv_change_30min"] = df["implied_volatility"].diff(30)
    df["option_volume_mean_30min"] = df["volume"].rolling(30).mean()
    df["delta_change_30min"] = df["corrected_delta"].diff(30)
    df["gamma_change_30min"] = df["gamma"].diff(30)

    # Cross & Interaction Features
    df["underlying_move_x_premium_move_15min"] = df["underlying_pct_15min"] * df["premium_ratio_15min"]
    df["underlying_move_x_iv_change_15min"] = df["underlying_pct_15min"] * df["iv_change_30min"]
    df["volume_surge_x_premium_spike"] = df["volume_surge"] * df["premium_spike_3min"]
    df["underlying_volume_x_premium_change"] = df["underlying_volume_mean_30min"] * df["premium_return_30min"]
    df["underlying_volatility_x_iv"] = df["underlying_volatility_30min"] * df["implied_volatility"]
    df["gamma_x_underlying_volatility"] = df["gamma"] * df["underlying_volatility_30min"]
    df["delta_x_underlying_momentum"] = df["corrected_delta"] * df["underlying_return_30min"]
    df["session_morning_x_volatility"] = df["session_morning"] * df["volatility"]
    df["session_afternoon_x_volume"] = df["session_afternoon"] * df["volume"]

    # ────────────────────────────────────────────────────────────────
    # 🚦 Market Regime Features
    # ────────────────────────────────────────────────────────────────

    # ATR-based volatility regime (20-period ATR using underlying data)
    df['underlying_atr_mean_20'] = df['underlying_atr'].rolling(20).mean()

    df['high_volatility_regime'] = (df['underlying_atr'] > df['underlying_atr_mean_20'] * 1.2).astype(int)
    df['low_volatility_regime'] = (df['underlying_atr'] < df['underlying_atr_mean_20'] * 0.8).astype(int)

    # Sideways (range-bound) regime: Narrow Bollinger Bands
    bollinger_bandwidth_mean = df['bollinger_bandwidth'].rolling(20).mean()
    df['sideways_regime'] = (df['bollinger_bandwidth'] < bollinger_bandwidth_mean * 0.75).astype(int)

    # Trending regime: Strong ADX (>25) and slope positive
    df['trending_regime'] = ((df['adx'] > 25) & (df['adx_slope'] > 0)).astype(int)

    # ────────────────────────────────────────────────────────────────
    # 📈 Open Interest Built-up Analysis (Underlying & Options)
    # ────────────────────────────────────────────────────────────────

    # Underlying OI and price changes
    df['underlying_oi_change'] = df['underlying_oi'].diff()
    df['underlying_price_change'] = df['underlying_close'].diff()

    df['underlying_long_buildup'] = (
            (df['underlying_oi_change'] > 0) & (df['underlying_price_change'] > 0)
    ).astype(int)

    df['underlying_short_buildup'] = (
            (df['underlying_oi_change'] > 0) & (df['underlying_price_change'] < 0)
    ).astype(int)

    df['underlying_long_unwinding'] = (
            (df['underlying_oi_change'] < 0) & (df['underlying_price_change'] < 0)
    ).astype(int)

    df['underlying_short_covering'] = (
            (df['underlying_oi_change'] < 0) & (df['underlying_price_change'] > 0)
    ).astype(int)

    # Option OI and Premium changes
    df['option_oi_change'] = df['open_interest'].diff()
    df['option_premium_change'] = df['close'].diff()

    df['option_long_buildup'] = (
            (df['option_oi_change'] > 0) & (df['option_premium_change'] > 0)
    ).astype(int)

    df['option_short_buildup'] = (
            (df['option_oi_change'] > 0) & (df['option_premium_change'] < 0)
    ).astype(int)

    df['option_long_unwinding'] = (
            (df['option_oi_change'] < 0) & (df['option_premium_change'] < 0)
    ).astype(int)

    df['option_short_covering'] = (
            (df['option_oi_change'] < 0) & (df['option_premium_change'] > 0)
    ).astype(int)

    # ────────────────────────────────────────────────────────────────
    # 📏 Distance to Reference Points
    # ────────────────────────────────────────────────────────────────

    # Distance to Previous Day's High and Low (Underlying)
    df['dist_to_prev_day_high'] = (df['underlying_close'] - df['prev_day_high']) / df['prev_day_high']
    df['dist_to_prev_day_low'] = (df['underlying_close'] - df['prev_day_low']) / df['prev_day_low']

    # Distance to Pivot Points
    df['dist_to_pivot'] = (df['underlying_close'] - df['prev_day_pivot']) / df['prev_day_pivot']
    df['dist_to_r1'] = (df['underlying_close'] - df['prev_day_r1']) / df['prev_day_r1']
    df['dist_to_s1'] = (df['underlying_close'] - df['prev_day_s1']) / df['prev_day_s1']

    # Distance to Opening Range High/Low
    df['dist_to_opening_range_high'] = (df['underlying_close'] - df['opening_range_high']) / df['opening_range_high']
    df['dist_to_opening_range_low'] = (df['underlying_close'] - df['opening_range_low']) / df['opening_range_low']

    # ────────────────────────────────────────────────────────────────
    # 🎯 Cross & Interaction Features (Additional Combinations)
    # ────────────────────────────────────────────────────────────────

    # Trending regime combined with built-up indicators
    df['trending_x_long_buildup'] = df['trending_regime'] * df['underlying_long_buildup']
    df['trending_x_short_buildup'] = df['trending_regime'] * df['underlying_short_buildup']

    # Sideways regime combined with short-covering signals
    df['sideways_x_short_covering'] = df['sideways_regime'] * df['underlying_short_covering']

    # High volatility combined with option long buildup
    df['high_vol_x_option_long_buildup'] = df['high_volatility_regime'] * df['option_long_buildup']

    # Low volatility combined with premium IV drop
    df['low_vol_x_iv_drop'] = df['low_volatility_regime'] * (df['iv_change_30min'] < 0).astype(int)

    # Distance to pivot × trending regime (potential breakouts)
    df['dist_pivot_x_trending'] = df['dist_to_pivot'] * df['trending_regime']

    # ────────────────────────────────────────────────────────────────
    # 🔥 Regime Aggregated Score (Simplified Market Condition Score)
    # ────────────────────────────────────────────────────────────────

    # Combine multiple conditions into single numeric regime indicators
    df['regime_score'] = (
            df['high_volatility_regime']
            + df['trending_regime']
            - df['sideways_regime']
    ).clip(-1, 2)

    # ────────────────────────────────────────────────────────────────
    # ✅ Final boolean dtype enforcement for regime/buildup indicators
    # ────────────────────────────────────────────────────────────────

    bool_columns = [
        'high_volatility_regime', 'low_volatility_regime', 'sideways_regime', 'trending_regime',
        'underlying_long_buildup', 'underlying_short_buildup', 'underlying_long_unwinding',
        'underlying_short_covering', 'option_long_buildup', 'option_short_buildup',
        'option_long_unwinding', 'option_short_covering'
    ]

    df[bool_columns] = df[bool_columns].astype(int)

    # Ichimoku Cloud
    ichi = ta.trend.IchimokuIndicator(df["underlying_high"], df["underlying_low"])
    df["u_ichimoku_a"] = ichi.ichimoku_a()
    df["u_ichimoku_b"] = ichi.ichimoku_b()

    # Aroon
    # Aroon (requires high & low)
    aroon = ta.trend.AroonIndicator(
        high=df["underlying_high"],
        low=df["underlying_low"],
        window=14
    )
    df["u_aroon_up"] = aroon.aroon_up()
    df["u_aroon_down"] = aroon.aroon_down()

    # KST Oscillator
    df["u_kst"] = ta.trend.kst(df["underlying_close"])

    # Parabolic SAR
    psar = ta.trend.PSARIndicator(
        high=df["underlying_high"],
        low=df["underlying_low"],
        close=df["underlying_close"],
        step=0.02,  # default acceleration factor
        max_step=0.2  # default maximum
    )
    df["u_psar"] = psar.psar()

    # 1) compute end-of-week highs & lows
    weekly = (
        df.set_index("date")
            .resample("W")  # one row per weekly bin
            .agg(weekly_high=("underlying_high", "max"),
                 weekly_low=("underlying_low", "min"))
            .shift(1)  # move everything back one week
    )

    df = df.set_index("date").merge(weekly, how="left", left_index=True, right_index=True)
    df[['weekly_high', 'weekly_low']] = df[['weekly_high', 'weekly_low']].ffill().reset_index(drop=True)
    df.reset_index(inplace=True)

    # 3) now distance features are truly “versus last week,” no leakage
    df["dist_weekly_high"] = (df["underlying_close"] - df["weekly_high"]) / df["weekly_high"]
    df["dist_weekly_low"] = (df["underlying_close"] - df["weekly_low"]) / df["weekly_low"]

    df["high_vol_x_trending"] = df["high_volatility_regime"] * df["trending_regime"]
    df["sideways_x_long_buildup"] = df["sideways_regime"] * df["underlying_long_buildup"]

    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def compute_intraday_oi_sentiment(df_all_symbols, strike_diff, strike_window=20, top_n_levels=5):
    df = df_all_symbols.copy()
    df = df.sort_values('date')
    # Step 1: Determine ATM strike rounded to nearest 50 per timestamp
    atm_by_time = (df.groupby('date')['underlying_close'].mean() / strike_diff).round() * strike_diff
    df = df.merge(atm_by_time.rename('atm_strike'), on='date')

    # Step 2: Filter to nearby strikes (±strike_window)
    df['strike'] = df['strike'].astype(float)
    df['strike_diff'] = (df['strike'] - df['atm_strike']) / strike_diff
    df = df[df['strike_diff'].abs() <= strike_window]

    # Get nearest expiry per day
    nearest_expiry = df.groupby('date_only')['expiry_date'].min().reset_index().rename(
        columns={'expiry_date': 'nearest_expiry_date'})
    df = df.merge(nearest_expiry, on='date_only', how='left')
    df_nearest_expiry = df[df['expiry_date'] == df['nearest_expiry_date']].drop(columns=['nearest_expiry_date'])

    result_rows = []

    # Ensure all timestamps are timezone-naive for safe comparison
    df_nearest_expiry['date'] = pd.to_datetime(df_nearest_expiry['date']).dt.tz_localize(None)
    for ts, group in df_nearest_expiry.groupby('date'):
        
        if ts > pd.to_datetime('2025-06-26 15:29:00'):
            pass
        atm = group['atm_strike'].iloc[0]

        # Group CE/PE by strike
        ce_oi = group[group['option_type'] == 'CE'].groupby('strike')['open_interest'].sum()
        pe_oi = group[group['option_type'] == 'PE'].groupby('strike')['open_interest'].sum()

        total_ce = ce_oi.sum()
        total_pe = pe_oi.sum()
        pcr = total_pe / (total_ce + 1e-6)

        result = {
            'date': ts,
            'pcr': pcr,
        }

        # Pre-fill resistance and support keys with default values
        for i in range(1, top_n_levels + 1):
            result[f'resistance_{i}'] = 0
            result[f'resistance_distance_{i}'] = 0
            result[f'resistance_oi_{i}'] = 0
            result[f'resistance_strength_{i}'] = 0.0

            result[f'support_{i}'] = 0
            result[f'support_distance_{i}'] = 0
            result[f'support_oi_{i}'] = 0
            result[f'support_strength_{i}'] = 0.0

        # -----------------------
        # ✅ Resistance Logic (Above ATM)
        # -----------------------
        resistance_levels = []
        prev_oi = ce_oi.get(atm, 0)  # Start comparing with ATM OI
        for strike in sorted(ce_oi.index):
            if strike <= atm:
                continue
            oi = ce_oi[strike]
            # Only add if OI is strictly higher than previous level
            if oi > prev_oi:
                resistance_levels.append((strike, oi))
                prev_oi = oi  # Update reference for next level
            if len(resistance_levels) == top_n_levels:
                break

        for i, (strike, oi) in enumerate(resistance_levels, start=1):
            distance = strike - atm
            strength = oi / (total_ce + 1e-6)
            result[f'resistance_{i}'] = strike
            result[f'resistance_distance_{i}'] = distance
            result[f'resistance_oi_{i}'] = oi
            result[f'resistance_strength_{i}'] = round(strength, 4)

        # -----------------------
        # ✅ Support Logic (Below ATM)
        # -----------------------
        support_levels = []
        prev_oi = pe_oi.get(atm, 0)
        for strike in sorted(pe_oi.index, reverse=True):
            if strike >= atm:
                continue
            oi = pe_oi[strike]
            # Only add if OI is strictly higher than previous level
            if oi > prev_oi:
                support_levels.append((strike, oi))
                prev_oi = oi  # Update reference for next level
            if len(support_levels) == top_n_levels:
                break

        for i, (strike, oi) in enumerate(support_levels, start=1):
            distance = atm - strike
            strength = oi / (total_pe + 1e-6)
            result[f'support_{i}'] = strike
            result[f'support_distance_{i}'] = distance
            result[f'support_oi_{i}'] = oi
            result[f'support_strength_{i}'] = round(strength, 4)

        result_rows.append(result)

    return pd.DataFrame(result_rows)


def process_symbol(symbol, start_datetime, end_datetime, symbol_df, pcr_support_df, stop_loss_percent, target_percent):
    # logging.info(f"🔹 Processing {symbol} for {start_datetime[:10]}")
    df = symbol_df.copy()

    # ✅ Merge PCR, support, resistance info
    df = df.merge(pcr_support_df, on='date', how='left')
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = label_trades(df, stop_loss_percent, target_percent)
    df = supertrend(df)
    df = calculate_fibonacci(df)
    df = calculate_features(df)
    # df = indicator_main(df)
    # print(f"✅ Processed {symbol} with {len(df)} rows")
    return df


def save_monthly_parquet(df, base_dir, date_column='date'):
    """
    Saves dataframe into monthly partitioned parquet files based on the specified date column.

    Args:`
        df (DataFrame): Pandas DataFrame containing the data to be saved.
        base_dir (str): Base directory where monthly parquet files will be stored.
        date_column (str): Column name containing datetime information.
    """

    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Extract year and month for file naming
    df['year_month'] = df[date_column].dt.strftime('%Y_%m')

    # Group by year and month, save separately
    for year_month, group_df in df.groupby('year_month'):
        # month_dir = os.path.join(base_dir, f"month={year_month}")
        os.makedirs(base_dir, exist_ok=True)

        file_path = os.path.join(base_dir, f"data_{year_month}.parquet")

        # Save parquet file with zstd compression
        group_df.drop(columns=['year_month']).to_parquet(
            file_path,
            compression='zstd',
            index=False
        )

        print(f"✅ Saved data for {year_month} to {file_path}")


def fe_main(symbol=None, df_nifty=None, df_option=None, df_vix=None, start_date=None, strike_diff=50,
            stop_loss_percent=10, target_percent=20, cpu_cores=5):
    # Get all unique months from df_nifty['date'] (or df_options_month if you prefer)
    trade_month = pd.to_datetime(df_nifty['date']).dt.to_period('M').dt.to_timestamp()
    months_to_process = sorted(trade_month.drop_duplicates())

    # how much history you need to fully warm up every rolling window?
    # e.g. if your longest lookback is 200 periods at 15-min, that's 200*15min = 3000min ≈ 2 days.
    # you could be conservative and grab 10 days:
    LOOKBACK_DAYS = 15
    for month_start_date in tqdm(months_to_process, desc="Processing Months"):
        month_end_date = (month_start_date + pd.DateOffset(months=1)) - pd.Timedelta(minutes=1)
        current_start = pd.to_datetime((month_start_date - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d 09:15:00"))
        current_end = pd.to_datetime(f"{month_end_date.strftime('%Y-%m-%d')} 15:30:00")

        # ✅ Fetch Nifty data for the current month
        df_underlying_month = df_nifty[(df_nifty['date'] >= current_start) & (df_nifty['date'] <= current_end)]

        # ✅ Fetch VIX data for the current month
        df_vix_month = df_vix[(df_vix['date'] >= current_start) & (df_vix['date'] <= current_end)]

        # ✅ Fetch ALL options data for the current month at once
        df_options_month = df_option[(df_option['date'] >= current_start) & (df_option['date'] <= current_end)]

        # ✅ Merge Nifty and VIX data ONCE PER MONTH
        monthly_merged_df = df_options_month.merge(df_underlying_month, on="date", how="left") \
            .merge(df_vix_month, on="date", how="left")
        monthly_merged_df.dropna(subset=["underlying_close", "vix_close"], inplace=True)
        del df_underlying_month, df_vix_month, df_options_month
        if monthly_merged_df.empty:
            continue
        monthly_merged_df['date_only'] = pd.to_datetime(monthly_merged_df['date']).dt.date
        # ✅ Compute once per month
        pcr_support_df = compute_intraday_oi_sentiment(monthly_merged_df, strike_diff)

        # ✅ VERY IMPORTANT: Split dataframe ONCE here into symbol-specific chunks
        symbols = monthly_merged_df['symbol'].unique()
        symbol_groups = {
            symbol: data.copy()
            for symbol, data in monthly_merged_df.groupby("symbol")
        }
        results = []
        with Pool(cpu_cores, maxtasksperchild=10) as pool:
            args = [
                (symbol, current_start, current_end, symbol_groups[symbol], pcr_support_df, stop_loss_percent, target_percent)
                for symbol in symbols if symbol in symbol_groups
            ]
            # process_symbol(*args[0])
            results = list(tqdm(pool.starmap(process_symbol, args), total=len(args)))
            # for i in tqdm(args):
            #     result = process_symbol(*i)
            #     results.append(result)

        # # ✅ Store processed month's data efficiently
        if len(results) == 0:
            continue
        monthly_result_df = pd.concat(results)
        del results
        monthly_result_df.drop_duplicates(subset=["date", "symbol"], inplace=True)
        monthly_result_df.drop(columns=['date_only'], inplace=True, errors='ignore')
        true_start = f"{month_start_date.strftime('%Y-%m-%d')} 09:15:00"
        monthly_result_df = monthly_result_df[
            monthly_result_df['date'] >= true_start
            ].copy()
        base_dir = f'parquet_files/{symbol}'
        save_monthly_parquet(monthly_result_df, base_dir)

    print("✅ All months processed and stored successfully!")
