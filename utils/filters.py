import pandas as pd
from pathlib import Path


def filter_underlying_data(df: pd.DataFrame, days_to_expiry=30) -> pd.DataFrame:
    """
    Filters the DataFrame for a specific symbol and selects specified columns.

    Args:
        df (pd.DataFrame): The DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame with selected columns.
    """
    df = df.copy()
    # df['days_to_expiry'] = (df['expiry_date'] - df['date']).dt.days
    # df = df[(df['days_to_expiry'] <= days_to_expiry) & (df['days_to_expiry'] >= 2)]
    df = df[~df['symbol'].str.contains(r'\w+NXT\d+', case=False, na=False, regex=True)]
    columns = {
        'open': 'underlying_open',
        'high': 'underlying_high',
        'low': 'underlying_low',
        'close': 'underlying_close',
        'volume': 'underlying_volume',
        'open_interest': 'underlying_oi'
    }
    # Select and rename columns
    df = df.rename(columns=columns)
    selected_columns = ['date', 'underlying_open', 'underlying_high', 'underlying_low', 'underlying_close',
                        'underlying_volume', 'underlying_oi']
    df = df[selected_columns].sort_values('date')
    return df.reset_index(drop=True)


def filter_option_data(df: pd.DataFrame, days_to_expiry=45) -> pd.DataFrame:
    """
    Filters the DataFrame for options data and selects specified columns.

    Args:
        df (pd.DataFrame): The DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame with selected columns.
        :param days_to_expiry:
    """
    df = df.copy()
    # df['days_to_expiry'] = (pd.to_datetime(df['expiry_date']) - pd.to_datetime(df['date'])).dt.days
    # df = df[(df['days_to_expiry'] <= days_to_expiry) & (df['days_to_expiry'] >= 2)]

    df = df[~df['symbol'].str.contains(r'\w+NXT\d+', case=False, na=False, regex=True)]
    selected_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'strike', 'option_type',
                        'expiry_date', 'open_interest']

    # Filter for expiry_date <= date + 45 days
    df = df[pd.to_datetime(df['expiry_date']) <= pd.to_datetime(df['date']) + pd.Timedelta(days=days_to_expiry)]
    df = df[selected_columns]

    return df.reset_index(drop=True)


def load_data_train(pth, start_date: pd.Timestamp, cutoff_date: pd.Timestamp, strike_diff: float,
                    le_days_to_expiry: int, gt_days_to_expiry: int, ) -> pd.DataFrame:
    parquet_files = sorted(Path(pth).rglob("*.parquet"))
    df_list = []
    for pq_file in parquet_files:
        temp_df = pd.read_parquet(pq_file)
        # Filter immediately after loading each parquet file
        temp_df = temp_df[
            (temp_df['trade_outcome'] != 0) &
            (temp_df['atm_diff'] <= strike_diff / 2) &
            (temp_df['days_to_expiry'] <= le_days_to_expiry) &
            (temp_df['days_to_expiry'] > gt_days_to_expiry) &
            (temp_df['date'] <= cutoff_date) &
            (temp_df['date'] >= start_date)
            ]
        print(f"✅ Loaded {temp_df.shape[0]} rows, {temp_df.shape[1]} cols from {pq_file}")
        # print(', '.join([f"'{col}'" for col in temp_df.columns]))
        # Append only if filtered DataFrame isn't empty
        if not temp_df.empty:
            df_list.append(temp_df)
    # Concatenate after filtering
    df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} cols")

    selected_columns = [
        'open', 'high', 'low', 'close', 'volume', 'open_interest', 'underlying_open', 'underlying_high',
        'underlying_low', 'underlying_close', 'underlying_volume', 'underlying_oi', 'vix_open', 'vix_high', 'vix_low',
        'vix_close', 'pcr', 'resistance_1', 'resistance_distance_1', 'resistance_oi_1', 'resistance_strength_1',
        'support_1', 'support_distance_1', 'support_oi_1', 'support_strength_1', 'resistance_2',
        'resistance_distance_2', 'resistance_oi_2', 'resistance_strength_2', 'support_2', 'support_distance_2',
        'support_oi_2', 'support_strength_2', 'resistance_3', 'resistance_distance_3', 'resistance_oi_3',
        'resistance_strength_3', 'support_3', 'support_distance_3', 'support_oi_3', 'support_strength_3',
        'resistance_4', 'resistance_distance_4', 'resistance_oi_4', 'resistance_strength_4', 'support_4',
        'support_distance_4', 'support_oi_4', 'support_strength_4', 'resistance_5', 'resistance_distance_5',
        'resistance_oi_5', 'resistance_strength_5', 'support_5', 'support_distance_5', 'support_oi_5',
        'support_strength_5', 'supertrend', 'lower_band', 'upper_band', 'fib_high', 'fib_low', 'fib_23_6', 'fib_38_2',
        'fib_50', 'fib_61_8', 'fib_78_6', 'days_to_expiry', 'minutes_to_expiry', 'hour', 'day_of_week', 'atm_diff',
        'atr', 'underlying_atr', 'adx', 'u_adx', 'vwap', 'underlying_vwap', 'macd', 'macd_signal', 'macd_diff', 'rsi',
        'underlying_rsi', 'rsi_divergence', 'underlying_rsi_divergence', 'bollinger_high', 'bollinger_mid',
        'bollinger_low', 'underlying_bollinger_high', 'underlying_bollinger_mid', 'underlying_bollinger_low', 'sma_5',
        'sma_20', 'ema_9', 'ema_21', 'ema_crossover', 'underlying_sma_50', 'underlying_sma_200',
        'underlying_sma_crossover', 'intrinsic_value', 'moneyness', 'delta', 'gamma', 'theta', 'vix_change',
        'vix_volatility', 'vix_sma_50', 'vix_sma_200', 'vix_sma_crossover', 'rsi_slope', 'adx_slope',
        'bollinger_bandwidth', 'close_lag_1', 'rsi_lag_1', 'volume_lag_1', 'vwap_lag_1', 'delta_lag_1',
        'open_interest_lag_1', 'close_lag_2', 'rsi_lag_2', 'volume_lag_2', 'vwap_lag_2', 'delta_lag_2',
        'open_interest_lag_2', 'close_lag_3', 'rsi_lag_3', 'volume_lag_3', 'vwap_lag_3', 'delta_lag_3',
        'open_interest_lag_3', 'close_lag_4', 'rsi_lag_4', 'volume_lag_4', 'vwap_lag_4', 'delta_lag_4',
        'open_interest_lag_4', 'close_lag_5', 'rsi_lag_5', 'volume_lag_5', 'vwap_lag_5', 'delta_lag_5',
        'open_interest_lag_5', 'bullish_price', 'bullish_volume', 'bullish_buildup', 'bearish_price', 'bearish_volume',
        'bearish_buildup', 'price_up_1', 'volume_up_1', 'price_up_2', 'volume_up_2', 'price_up_3', 'volume_up_3',
        'obv_direction_1', 'obv_direction_2', 'obv_direction_3', 'bullish_score', 'cumulative_obv', 'rsi_trend_up',
        'bullish_confirmed', 'volatility', 'vol_q25_so_far', 'low_vol_bullish', 'weighted_bullish_score',
        'cumulative_day_volume', 'day_high_so_far', 'day_low_so_far', 'distance_from_lb', 'distance_to_hb',
        'distance_from_high', 'distance_from_low', 'volume_mean', 'volume_std', 'volume_zscore',
        'intraday_trend_strength', 'cumulative_delta', 'implied_volatility', 'vix_ma', 'iv_skew',
        'volatility_adjusted_volume', 'price_impact', 'gamma_exposure', 'session_progress', 'momentum_1',
        'momentum_2', 'momentum_3', 'close_per_1', 'close_per_2', 'close_per_3', 'close_per_4', 'close_per_5',
        'normalized_close_5', 'normalized_close_15', 'normalized_close_30', 'sin_time', 'cos_time', 'open_interest_5',
        'open_interest_15', 'open_interest_30', 'price_momentum', 'volatility_ratio', 'price_range', 'stoch_rsi',
        'price_to_vwap', 'volumexvwap', 'dist_to_fib_61_8', 'dist_to_fib_38_2', 'ewma_close_5', 'ewma_rsi_14', 'roc_5',
        'momentum_acceleration', 'last_bullish', 'time_since_bullish', 'volatility_spike', 'close_to_20_high',
        'close_to_20_low', 'rolling_return_5', 'rolling_return_15', 'oi_change', 'oi_to_volume_ratio', 'volume_surge',
        'price_expansion', 'vwap_zscore', 'deviation_from_high', 'deviation_from_low', 'price_spike', 'macd_slope',
        'rsi_macd_divergence', 'price_to_atr', 'daily_underlying_open', 'prev_day_close', 'prev_day_open',
        'prev_day_high', 'prev_day_low', 'prev_day_vix_close', 'gap_pct', 'underlying_pct_from_prev_close',
        'vix_change_from_prev_day', 'gap_up', 'gap_down', 'vix_spike', 'bias_pe_vix', 'combined_pe_bias',
        'combined_ce_bias', 'premium_ratio_3min', 'premium_ratio_5min', 'premium_ratio_15min', 'underlying_pct_3min',
        'underlying_pct_5min', 'underlying_pct_15min', 'underlying_move_vs_atr', 'underlying_hl_range',
        'underlying_range_ratio', 'underlying_break_high_20', 'underlying_break_low_20', 'premium_volatility_spike',
        'corrected_delta', 'gamma_explosion', 'market_open_volatility', 'afternoon_volatility', 'double_premium',
        'triple_premium', 'premium_spike_1min', 'premium_spike_3min', 'premium_spike_5min', 'underlying_up_0_5pct',
        'underlying_up_1pct', 'underlying_up_1_5pct', 'premium_underlying_elasticity', 'premium_volume_spike',
        'atm_strike_distance_pct', 'atm_proximity', 'delta_change_3min', 'gamma_change_3min', 'rapid_gamma_spike',
        'underlying_hammer', 'underlying_bullish_engulfing', 'underlying_bearish_engulfing', 'premium_intrinsic_ratio',
        'underlying_rapid_drop', 'vix_rapid_jump', 'high_profit_trade', 'premium_volatility_rank',
        'underlying_pct_from_open', 'underlying_distance_points_from_open',
        'underlying_distance_points_from_prev_close',
        'underlying_high_since_open', 'underlying_low_since_open', 'distance_from_open_high', 'distance_from_open_low',
        'underlying_return_3min', 'underlying_return_5min', 'is_ce', 'is_pe', 'directional_alignment', 'regime_flag',
        'sample_weight', 'momentum_x_volatility', 'vwap_vs_rsi', 'delta_x_gamma', 'candle_body_pct', 'candle_range_pct',
        'upper_wick', 'lower_wick', 'body_to_range_ratio', 'volume_vs_5', 'volume_vs_10', 'strong_bullish_candle',
        'strong_bearish_candle', 'rejection_candle', 'candle_body_pct_lag_1', 'candle_body_pct_lag_2',
        'candle_range_pct_lag_1', 'candle_range_pct_lag_2', 'upper_wick_lag_1', 'upper_wick_lag_2', 'lower_wick_lag_1',
        'lower_wick_lag_2', 'body_to_range_ratio_lag_1', 'body_to_range_ratio_lag_2', 'volume_vs_5_lag_1',
        'volume_vs_5_lag_2', 'volume_vs_10_lag_1', 'volume_vs_10_lag_2', 'strong_bullish_candle_lag_1',
        'strong_bullish_candle_lag_2', 'strong_bearish_candle_lag_1', 'strong_bearish_candle_lag_2',
        'rejection_candle_lag_1', 'rejection_candle_lag_2', 'bias_ce', 'bias_pe', 'support_dominant_1',
        'resistance_dominant_1', 'support_score_1', 'resistance_score_1', 'support_dominant_2', 'resistance_dominant_2',
        'support_score_2', 'resistance_score_2', 'support_dominant_3', 'resistance_dominant_3', 'support_score_3',
        'resistance_score_3', 'support_strength', 'resistance_strength', 'oi_bias_ce', 'oi_bias_pe', 'near_support_1',
        'near_resistance_1', 'support_hold_ce', 'resistance_block_pe', 'final_ce_signal', 'vix_spike_pe',
        'final_pe_signal', 'rolling_range', 'rr_q25_so_far', 'consolidation_breakout', 'macd_signal_crossover',
        'rsi_overbought', 'rsi_oversold', 'signal_buy', 'strong_support_zone', 'strong_resistance_zone',
        'weak_support_zone', 'weak_resistance_zone', 'oi_support_dominant', 'oi_resistance_dominant', 'squeeze_zone',
        'support_strength_x_distance', 'resistance_strength_x_distance', 'delta_x_oi', 'vix_x_iv', 'close_x_gap',
        'support_x_t', 'resistance_x_t', 'bucket_start', 'underlying_15_open', 'underlying_15_high',
        'underlying_15_low', 'underlying_15_close', 'underlying_15_sma_20', 'underlying_15_sma_50',
        'underlying_15_sma_200', 'underlying_15_rsi', 'underlying_15_sma_crossover', 'underlying_15_trend_up',
        'underlying_15_rsi_trend', 'underlying_15_range', 'underlying_15_range_rank', 'htf_breakout_confirmed',
        'htf_breakdown_confirmed', 'htf_trend_confirmation', 'vwap_slope_3', 'vwap_slope_direction',
        'compression_score', 'compression_detected', 'cci', 'cci_breakout', 'cci_breakdown', 'green_streak',
        'red_streak', 'green_streak_len', 'red_streak_len', 'distance_to_vwap', 'mean_reversion_signal', 'time_bin',
        'pre_breakout_time', 'price_thrust', 'micro_pullback', 'breakout_base', 'leading_score', 'rs_ratio',
        'rs_ratio_signal', 'iv_decline', 'premium_spike_but_iv_down', 'ha_close', 'ha_trend_strength',
        'trend_and_squeeze', 'vix_divergence', 'fvg_up', 'fvg_down', 'smart_money_footprint', 'hidden_bullish',
        'hidden_bearish', 'volume_shock_reversal', 'close_below_half', 'rejection_upper_half', 'bull_strength',
        'bear_strength', 'wick_imbalance', 'elasticity_ratio', 'elasticity_failure', 'squeeze_exhaustion', 'lower_high',
        'higher_low', 'smart_price_action_trigger', 'daily_high_so_far', 'daily_low_so_far', 'daily_range',
        'underlying_day_high_so_far', 'underlying_day_low_so_far', 'underlying_daily_range',
        'underlying_distance_from_high', 'underlying_distance_from_low', 'atr_vs_range_ratio', 'range_squeeze_alert',
        'iv_vs_vix_diff', 'option_undervalued', 'option_overvalued', 'micro_breakout', 'micro_breakout_count',
        'expiry_rush', 'vwap_upper', 'vwap_lower', 'snap_reversal_up', 'snap_reversal_down', 'underlying_moved',
        'premium_did_not_move', 'mispricing_window', 'theta_crush_risk', 'explosive_entry_score', 'break_count',
        'consolidation_before_breakout', 'rsi_slope_5', 'oi_ce_change', 'smart_money_score', 'failed_breakout',
        'u_sma_20', 'u_sma_50', 'u_sma_200', 'u_ema_12', 'u_ema_26', 'u_ema_xover', 'u_macd', 'u_macd_signal',
        'u_macd_hist', 'u_trix', 'u_rsi', 'u_rsi_slope', 'u_stochrsi', 'u_roc_5', 'u_cci', 'u_willr', 'u_donch_high_20',
        'u_donch_low_20', 'u_kelt_ma', 'u_kelt_up', 'u_kelt_lo', 'u_bb_bw', 'u_obv', 'u_chaikin', 'u_pivot', 'u_r1',
        'u_s1', 'u_r2', 'u_s2', 'u_open_range_hi', 'u_open_range_lo', 'u_5m_sma', 'u_15m_sma', 'typ_price',
        'cum_tp_vol',
        'cum_vol', 'vwap_1m', 'dist_from_vwap', 'prev_day_pivot', 'prev_day_r1', 'prev_day_s1', 'prev_day_r2',
        'prev_day_s2', 'u_ret_1m', 'u_price_diff', 'oi_change_1m', 'oi_to_vol', 'oi_vol_ratio_5', 'u_mfi',
        'u_return_5m', 'momentum_score', 'rolling_oi_delta_5m', 'volume_oi_ratio_5m', 'u_1h_sma_20', 'u_1h_trend_up',
        'underlying_15m_bb_breakout', 'vwap_slope_5', 'vwap_slope_reversal', 'opening_range_high', 'opening_range_low',
        'orb_breakout', 'orb_breakdown', 'minute_of_day', 'session_morning', 'session_midday', 'session_afternoon',
        'underlying_momentum_5m', 'volatility_compression_rank', 'underlying_range_expansion_5m',
        'momentum_range_combo',
        'u_distance_to_vwap', 'u_near_vwap_zone', 'u_vwap_slope_5', 'u_mean_reversion_signal', 'high_atr_regime',
        'low_atr_regime', 'range_breakout', 'break_prev_day_high', 'break_prev_day_low', 'gap_open_strength',
        'cumulative_return_session', 'short_term_pullback', 'compression_breakout', 'vwap_reclaim', 'atr_spike',
        'smart_money_flow', 'underlying_return_30min', 'underlying_return_60min', 'underlying_volatility_30min',
        'underlying_volatility_60min', 'underlying_volume_mean_30min', 'underlying_oi_change_30min',
        'underlying_position_vs_30min_high', 'underlying_position_vs_30min_low', 'premium_return_30min',
        'premium_return_60min', 'iv_change_30min', 'option_volume_mean_30min', 'delta_change_30min',
        'gamma_change_30min',
        'underlying_move_x_premium_move_15min', 'underlying_move_x_iv_change_15min', 'volume_surge_x_premium_spike',
        'underlying_volume_x_premium_change', 'underlying_volatility_x_iv', 'gamma_x_underlying_volatility',
        'delta_x_underlying_momentum', 'session_morning_x_volatility', 'session_afternoon_x_volume',
        'underlying_atr_mean_20', 'high_volatility_regime', 'low_volatility_regime', 'sideways_regime',
        'trending_regime', 'underlying_oi_change', 'underlying_price_change', 'underlying_long_buildup',
        'underlying_short_buildup', 'underlying_long_unwinding', 'underlying_short_covering', 'option_oi_change',
        'option_premium_change', 'option_long_buildup', 'option_short_buildup', 'option_long_unwinding',
        'option_short_covering', 'dist_to_prev_day_high', 'dist_to_prev_day_low', 'dist_to_pivot', 'dist_to_r1',
        'dist_to_s1', 'dist_to_opening_range_high', 'dist_to_opening_range_low', 'trending_x_long_buildup',
        'trending_x_short_buildup', 'sideways_x_short_covering', 'high_vol_x_option_long_buildup', 'low_vol_x_iv_drop',
        'dist_pivot_x_trending', 'regime_score', 'u_ichimoku_a', 'u_ichimoku_b', 'u_aroon_up', 'u_aroon_down',
        'u_kst', 'u_psar', 'weekly_high', 'weekly_low', 'dist_weekly_high', 'dist_weekly_low', 'high_vol_x_trending',
        'sideways_x_long_buildup',
        # Target & Timestamp
        'trade_outcome', 'date', 'option_type'
    ]

    # Verify columns exist
    selected_columns = [col for col in selected_columns if col in df.columns]
    df = df[selected_columns]
    print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_data_backtest(pth, test_start_date, strike_diff, le_days_to_expiry, gt_days_to_expiry) -> pd.DataFrame:
    parquet_files = sorted(Path(pth).rglob("*.parquet"))

    df_list = []

    for pq_file in parquet_files:
        temp_df = pd.read_parquet(pq_file)
        # Filter immediately after loading each parquet file
        temp_df = temp_df[
            (temp_df['days_to_expiry'] <= le_days_to_expiry+30) &
            (temp_df['date'] >= test_start_date) &
            (temp_df['atm_diff'] < strike_diff * 10) #&
            # (temp_df['days_to_expiry'] >= gt_days_to_expiry)
            ]
        print(f"✅ Loaded {temp_df.shape[0]} rows, {temp_df.shape[1]} cols from {pq_file}")
        # print(', '.join([f"'{col}'" for col in temp_df.columns]))
        # Append only if filtered DataFrame isn't empty
        if not temp_df.empty:
            df_list.append(temp_df)
    # Concatenate after filtering
    df = pd.concat(df_list, ignore_index=True)

    selected_columns = [
        'open', 'high', 'low', 'close', 'volume', 'open_interest', 'underlying_open', 'underlying_high',
        'underlying_low', 'underlying_close', 'underlying_volume', 'underlying_oi', 'vix_open', 'vix_high', 'vix_low',
        'vix_close', 'pcr', 'resistance_1', 'resistance_distance_1', 'resistance_oi_1', 'resistance_strength_1',
        'support_1', 'support_distance_1', 'support_oi_1', 'support_strength_1', 'resistance_2',
        'resistance_distance_2', 'resistance_oi_2', 'resistance_strength_2', 'support_2', 'support_distance_2',
        'support_oi_2', 'support_strength_2', 'resistance_3', 'resistance_distance_3', 'resistance_oi_3',
        'resistance_strength_3', 'support_3', 'support_distance_3', 'support_oi_3', 'support_strength_3',
        'resistance_4', 'resistance_distance_4', 'resistance_oi_4', 'resistance_strength_4', 'support_4',
        'support_distance_4', 'support_oi_4', 'support_strength_4', 'resistance_5', 'resistance_distance_5',
        'resistance_oi_5', 'resistance_strength_5', 'support_5', 'support_distance_5', 'support_oi_5',
        'support_strength_5', 'supertrend', 'lower_band', 'upper_band', 'fib_high', 'fib_low', 'fib_23_6', 'fib_38_2',
        'fib_50', 'fib_61_8', 'fib_78_6', 'days_to_expiry', 'minutes_to_expiry', 'hour', 'day_of_week', 'atm_diff',
        'atr', 'underlying_atr', 'adx', 'u_adx', 'vwap', 'underlying_vwap', 'macd', 'macd_signal', 'macd_diff', 'rsi',
        'underlying_rsi', 'rsi_divergence', 'underlying_rsi_divergence', 'bollinger_high', 'bollinger_mid',
        'bollinger_low', 'underlying_bollinger_high', 'underlying_bollinger_mid', 'underlying_bollinger_low', 'sma_5',
        'sma_20', 'ema_9', 'ema_21', 'ema_crossover', 'underlying_sma_50', 'underlying_sma_200',
        'underlying_sma_crossover', 'intrinsic_value', 'moneyness', 'delta', 'gamma', 'theta', 'vix_change',
        'vix_volatility', 'vix_sma_50', 'vix_sma_200', 'vix_sma_crossover', 'rsi_slope', 'adx_slope',
        'bollinger_bandwidth', 'close_lag_1', 'rsi_lag_1', 'volume_lag_1', 'vwap_lag_1', 'delta_lag_1',
        'open_interest_lag_1', 'close_lag_2', 'rsi_lag_2', 'volume_lag_2', 'vwap_lag_2', 'delta_lag_2',
        'open_interest_lag_2', 'close_lag_3', 'rsi_lag_3', 'volume_lag_3', 'vwap_lag_3', 'delta_lag_3',
        'open_interest_lag_3', 'close_lag_4', 'rsi_lag_4', 'volume_lag_4', 'vwap_lag_4', 'delta_lag_4',
        'open_interest_lag_4', 'close_lag_5', 'rsi_lag_5', 'volume_lag_5', 'vwap_lag_5', 'delta_lag_5',
        'open_interest_lag_5', 'bullish_price', 'bullish_volume', 'bullish_buildup', 'bearish_price', 'bearish_volume',
        'bearish_buildup', 'price_up_1', 'volume_up_1', 'price_up_2', 'volume_up_2', 'price_up_3', 'volume_up_3',
        'obv_direction_1', 'obv_direction_2', 'obv_direction_3', 'bullish_score', 'cumulative_obv', 'rsi_trend_up',
        'bullish_confirmed', 'volatility', 'vol_q25_so_far', 'low_vol_bullish', 'weighted_bullish_score',
        'cumulative_day_volume', 'day_high_so_far', 'day_low_so_far', 'distance_from_lb', 'distance_to_hb',
        'distance_from_high', 'distance_from_low', 'volume_mean', 'volume_std', 'volume_zscore',
        'intraday_trend_strength', 'cumulative_delta', 'implied_volatility', 'vix_ma', 'iv_skew',
        'volatility_adjusted_volume', 'price_impact', 'gamma_exposure', 'session_progress', 'momentum_1',
        'momentum_2', 'momentum_3', 'close_per_1', 'close_per_2', 'close_per_3', 'close_per_4', 'close_per_5',
        'normalized_close_5', 'normalized_close_15', 'normalized_close_30', 'sin_time', 'cos_time', 'open_interest_5',
        'open_interest_15', 'open_interest_30', 'price_momentum', 'volatility_ratio', 'price_range', 'stoch_rsi',
        'price_to_vwap', 'volumexvwap', 'dist_to_fib_61_8', 'dist_to_fib_38_2', 'ewma_close_5', 'ewma_rsi_14', 'roc_5',
        'momentum_acceleration', 'last_bullish', 'time_since_bullish', 'volatility_spike', 'close_to_20_high',
        'close_to_20_low', 'rolling_return_5', 'rolling_return_15', 'oi_change', 'oi_to_volume_ratio', 'volume_surge',
        'price_expansion', 'vwap_zscore', 'deviation_from_high', 'deviation_from_low', 'price_spike', 'macd_slope',
        'rsi_macd_divergence', 'price_to_atr', 'daily_underlying_open', 'prev_day_close', 'prev_day_open',
        'prev_day_high', 'prev_day_low', 'prev_day_vix_close', 'gap_pct', 'underlying_pct_from_prev_close',
        'vix_change_from_prev_day', 'gap_up', 'gap_down', 'vix_spike', 'bias_pe_vix', 'combined_pe_bias',
        'combined_ce_bias', 'premium_ratio_3min', 'premium_ratio_5min', 'premium_ratio_15min', 'underlying_pct_3min',
        'underlying_pct_5min', 'underlying_pct_15min', 'underlying_move_vs_atr', 'underlying_hl_range',
        'underlying_range_ratio', 'underlying_break_high_20', 'underlying_break_low_20', 'premium_volatility_spike',
        'corrected_delta', 'gamma_explosion', 'market_open_volatility', 'afternoon_volatility', 'double_premium',
        'triple_premium', 'premium_spike_1min', 'premium_spike_3min', 'premium_spike_5min', 'underlying_up_0_5pct',
        'underlying_up_1pct', 'underlying_up_1_5pct', 'premium_underlying_elasticity', 'premium_volume_spike',
        'atm_strike_distance_pct', 'atm_proximity', 'delta_change_3min', 'gamma_change_3min', 'rapid_gamma_spike',
        'underlying_hammer', 'underlying_bullish_engulfing', 'underlying_bearish_engulfing', 'premium_intrinsic_ratio',
        'underlying_rapid_drop', 'vix_rapid_jump', 'high_profit_trade', 'premium_volatility_rank',
        'underlying_pct_from_open', 'underlying_distance_points_from_open','underlying_distance_points_from_prev_close',
        'underlying_high_since_open', 'underlying_low_since_open', 'distance_from_open_high', 'distance_from_open_low',
        'underlying_return_3min', 'underlying_return_5min', 'is_ce', 'is_pe', 'directional_alignment', 'regime_flag',
        'sample_weight', 'momentum_x_volatility', 'vwap_vs_rsi', 'delta_x_gamma', 'candle_body_pct', 'candle_range_pct',
        'upper_wick', 'lower_wick', 'body_to_range_ratio', 'volume_vs_5', 'volume_vs_10', 'strong_bullish_candle',
        'strong_bearish_candle', 'rejection_candle', 'candle_body_pct_lag_1', 'candle_body_pct_lag_2',
        'candle_range_pct_lag_1', 'candle_range_pct_lag_2', 'upper_wick_lag_1', 'upper_wick_lag_2', 'lower_wick_lag_1',
        'lower_wick_lag_2', 'body_to_range_ratio_lag_1', 'body_to_range_ratio_lag_2', 'volume_vs_5_lag_1',
        'volume_vs_5_lag_2', 'volume_vs_10_lag_1', 'volume_vs_10_lag_2', 'strong_bullish_candle_lag_1',
        'strong_bullish_candle_lag_2', 'strong_bearish_candle_lag_1', 'strong_bearish_candle_lag_2',
        'rejection_candle_lag_1', 'rejection_candle_lag_2', 'bias_ce', 'bias_pe', 'support_dominant_1',
        'resistance_dominant_1', 'support_score_1', 'resistance_score_1', 'support_dominant_2', 'resistance_dominant_2',
        'support_score_2', 'resistance_score_2', 'support_dominant_3', 'resistance_dominant_3', 'support_score_3',
        'resistance_score_3', 'support_strength', 'resistance_strength', 'oi_bias_ce', 'oi_bias_pe', 'near_support_1',
        'near_resistance_1', 'support_hold_ce', 'resistance_block_pe', 'final_ce_signal', 'vix_spike_pe',
        'final_pe_signal', 'rolling_range', 'rr_q25_so_far', 'consolidation_breakout', 'macd_signal_crossover',
        'rsi_overbought', 'rsi_oversold', 'signal_buy', 'strong_support_zone', 'strong_resistance_zone',
        'weak_support_zone', 'weak_resistance_zone', 'oi_support_dominant', 'oi_resistance_dominant', 'squeeze_zone',
        'support_strength_x_distance', 'resistance_strength_x_distance', 'delta_x_oi', 'vix_x_iv', 'close_x_gap',
        'support_x_t', 'resistance_x_t', 'bucket_start', 'underlying_15_open', 'underlying_15_high',
        'underlying_15_low', 'underlying_15_close', 'underlying_15_sma_20', 'underlying_15_sma_50',
        'underlying_15_sma_200', 'underlying_15_rsi', 'underlying_15_sma_crossover', 'underlying_15_trend_up',
        'underlying_15_rsi_trend', 'underlying_15_range', 'underlying_15_range_rank', 'htf_breakout_confirmed',
        'htf_breakdown_confirmed', 'htf_trend_confirmation', 'vwap_slope_3', 'vwap_slope_direction',
        'compression_score', 'compression_detected', 'cci', 'cci_breakout', 'cci_breakdown', 'green_streak',
        'red_streak', 'green_streak_len', 'red_streak_len', 'distance_to_vwap', 'mean_reversion_signal', 'time_bin',
        'pre_breakout_time', 'price_thrust', 'micro_pullback', 'breakout_base', 'leading_score', 'rs_ratio',
        'rs_ratio_signal', 'iv_decline', 'premium_spike_but_iv_down', 'ha_close', 'ha_trend_strength',
        'trend_and_squeeze', 'vix_divergence', 'fvg_up', 'fvg_down', 'smart_money_footprint', 'hidden_bullish',
        'hidden_bearish', 'volume_shock_reversal', 'close_below_half', 'rejection_upper_half', 'bull_strength',
        'bear_strength', 'wick_imbalance', 'elasticity_ratio', 'elasticity_failure', 'squeeze_exhaustion', 'lower_high',
        'higher_low', 'smart_price_action_trigger', 'daily_high_so_far', 'daily_low_so_far', 'daily_range',
        'underlying_day_high_so_far', 'underlying_day_low_so_far', 'underlying_daily_range',
        'underlying_distance_from_high', 'underlying_distance_from_low', 'atr_vs_range_ratio', 'range_squeeze_alert',
        'iv_vs_vix_diff', 'option_undervalued', 'option_overvalued', 'micro_breakout', 'micro_breakout_count',
        'expiry_rush', 'vwap_upper', 'vwap_lower', 'snap_reversal_up', 'snap_reversal_down', 'underlying_moved',
        'premium_did_not_move', 'mispricing_window', 'theta_crush_risk', 'explosive_entry_score', 'break_count',
        'consolidation_before_breakout', 'rsi_slope_5', 'oi_ce_change', 'smart_money_score', 'failed_breakout',
        'u_sma_20', 'u_sma_50', 'u_sma_200', 'u_ema_12', 'u_ema_26', 'u_ema_xover', 'u_macd', 'u_macd_signal',
        'u_macd_hist', 'u_trix', 'u_rsi', 'u_rsi_slope', 'u_stochrsi', 'u_roc_5', 'u_cci', 'u_willr', 'u_donch_high_20',
        'u_donch_low_20', 'u_kelt_ma', 'u_kelt_up', 'u_kelt_lo', 'u_bb_bw', 'u_obv', 'u_chaikin', 'u_pivot', 'u_r1',
        'u_s1', 'u_r2', 'u_s2', 'u_open_range_hi', 'u_open_range_lo', 'u_5m_sma', 'u_15m_sma', 'typ_price','cum_tp_vol',
        'cum_vol', 'vwap_1m', 'dist_from_vwap', 'prev_day_pivot', 'prev_day_r1', 'prev_day_s1', 'prev_day_r2',
        'prev_day_s2', 'u_ret_1m', 'u_price_diff', 'oi_change_1m', 'oi_to_vol', 'oi_vol_ratio_5', 'u_mfi',
        'u_return_5m', 'momentum_score', 'rolling_oi_delta_5m', 'volume_oi_ratio_5m', 'u_1h_sma_20', 'u_1h_trend_up',
        'underlying_15m_bb_breakout', 'vwap_slope_5', 'vwap_slope_reversal', 'opening_range_high', 'opening_range_low',
        'orb_breakout', 'orb_breakdown', 'minute_of_day', 'session_morning', 'session_midday', 'session_afternoon',
        'underlying_momentum_5m', 'volatility_compression_rank', 'underlying_range_expansion_5m','momentum_range_combo',
        'u_distance_to_vwap', 'u_near_vwap_zone', 'u_vwap_slope_5', 'u_mean_reversion_signal', 'high_atr_regime',
        'low_atr_regime', 'range_breakout', 'break_prev_day_high', 'break_prev_day_low', 'gap_open_strength',
        'cumulative_return_session', 'short_term_pullback', 'compression_breakout', 'vwap_reclaim', 'atr_spike',
        'smart_money_flow', 'underlying_return_30min', 'underlying_return_60min', 'underlying_volatility_30min',
        'underlying_volatility_60min', 'underlying_volume_mean_30min', 'underlying_oi_change_30min',
        'underlying_position_vs_30min_high', 'underlying_position_vs_30min_low', 'premium_return_30min',
        'premium_return_60min', 'iv_change_30min', 'option_volume_mean_30min', 'delta_change_30min','gamma_change_30min',
        'underlying_move_x_premium_move_15min', 'underlying_move_x_iv_change_15min', 'volume_surge_x_premium_spike',
        'underlying_volume_x_premium_change', 'underlying_volatility_x_iv', 'gamma_x_underlying_volatility',
        'delta_x_underlying_momentum', 'session_morning_x_volatility', 'session_afternoon_x_volume',
        'underlying_atr_mean_20', 'high_volatility_regime', 'low_volatility_regime', 'sideways_regime',
        'trending_regime', 'underlying_oi_change', 'underlying_price_change', 'underlying_long_buildup',
        'underlying_short_buildup', 'underlying_long_unwinding', 'underlying_short_covering', 'option_oi_change',
        'option_premium_change', 'option_long_buildup', 'option_short_buildup', 'option_long_unwinding',
        'option_short_covering', 'dist_to_prev_day_high', 'dist_to_prev_day_low', 'dist_to_pivot', 'dist_to_r1',
        'dist_to_s1', 'dist_to_opening_range_high', 'dist_to_opening_range_low', 'trending_x_long_buildup',
        'trending_x_short_buildup', 'sideways_x_short_covering', 'high_vol_x_option_long_buildup', 'low_vol_x_iv_drop',
        'dist_pivot_x_trending', 'regime_score', 'u_ichimoku_a', 'u_ichimoku_b', 'u_aroon_up', 'u_aroon_down',
        'u_kst', 'u_psar', 'weekly_high', 'weekly_low', 'dist_weekly_high', 'dist_weekly_low', 'high_vol_x_trending',
        'sideways_x_long_buildup',
        # Target
        'trade_outcome', 'date', 'exit_price', 'exit_time', 'symbol', 'strike', 'option_type'
    ]
    selected_columns = [col for col in selected_columns if col in df.columns]
    df = df[selected_columns]
    df['option_type'] = df['option_type'].map({"CE": 0, "PE": 1})
    print(f"✅ Final shape: {df.shape}")
    return df
