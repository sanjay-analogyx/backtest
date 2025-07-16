import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

from nifty_features_engineering_v2 import fe_main
from utils.backtest_model_parquet_stacking_v2 import bt_main
from model_train_parquet_cat_v2 import cat_main
from model_train_parquet_lgbm_v2 import lgbm_main
from model_train_parquet_xgb_v2 import xgb_main
from model_train_parquet_stacking_v2 import stacking_main
from utils.download import download_vix_from_s3, load_or_download_all_ohlc_parquet
from utils.filters import filter_option_data, load_data_train, load_data_backtest
import boto3


def process_symbol(row: Dict[str, Any]):
    """
    Process one symbol end-to-end: download data, feature-engineer,
    train models, and optionally backtest.
    """
    symbol = row['symbol']
    start_date = row['global_start']
    fe_start_date = row.get('fe_start_date', None)
    s3 = boto3.client('s3')
    bucket_name = 'analogyx-trading-data'
    symbol_df = load_or_download_all_ohlc_parquet(
        bucket="analogyx-trading-data",
        prefix="NFO/",
        start_date=start_date,
        cpu_cores=min(cpu_count() - 1, 8),
        symbol=symbol,
    )
    if row.get('skip_fe', 0) in [1, 0]:
        print(f"[{symbol}] loading data…")
        if fe_start_date:
            symbol_df = symbol_df[symbol_df['date']>fe_start_date]
        # 1) prepare underlying frame
        fut_df = symbol_df[symbol_df['symbol'].str.lower() == f"{symbol}-I.NFO".lower()].copy()
        fut_df.rename(columns={
            'open': 'underlying_open',
            'high': 'underlying_high',
            'low': 'underlying_low',
            'close': 'underlying_close',
            'volume': 'underlying_volume',
            'open_interest': 'underlying_oi'
        }, inplace=True)
        fut_df = fut_df[[
            'date', 'underlying_open', 'underlying_high',
            'underlying_low', 'underlying_close',
            'underlying_volume', 'underlying_oi'
        ]]
        fut_df['date'] = pd.to_datetime(fut_df['date']).dt.floor('min')

        # 2) prepare options
        option_df = symbol_df[symbol_df['option_type'].isin(['CE', 'PE'])].copy()
        opt = filter_option_data(option_df, days_to_expiry=45)
        opt['date'] = pd.to_datetime(opt['date']).dt.floor('min')
        opt['strike'] = opt['strike'].astype(float)
        opt = opt.sort_values('date')

        # 3) VIX
        vix_df = download_vix_from_s3('analogyx-trading-data', "GDF DATA/INDIA VIX.csv")
        vix_df = vix_df[vix_df['date'] >= start_date].sort_values('date')

        # free memory
        del symbol_df, option_df

        # 4) Feature engineering
        print(f"[{symbol}] feature engineering…")
        fe_main(
            symbol=symbol,
            df_nifty=fut_df,
            df_option=opt,
            df_vix=vix_df,
            start_date=start_date,
            strike_diff=row['strike_diff'],
            stop_loss_percent=row['stop_loss_percent'],
            target_percent=row['target_percent'],
            cpu_cores=min(cpu_count() - 1, 40)
        )
        del opt, vix_df, fut_df

    # 5) Training
    if row.get('train_model', 0) in [0]:
        print(f"[{symbol}] training…")
        parquet_pth = f'parquet_files/{symbol}'
        train_data = load_data_train(
            parquet_pth,
            start_date=pd.to_datetime(row['train_start_date']),
            cutoff_date=pd.to_datetime(row['cutoff_date']),
            strike_diff=row["strike_diff"],
            le_days_to_expiry=row['le_days_to_expiry'],
            gt_days_to_expiry=row['gt_days_to_expiry'],
        )
        xgb_model = xgb_main(df=train_data.copy(), symbol=symbol, split_date=row['split_date'])
        cat_model = cat_main(df=train_data.copy(), symbol=symbol, split_date=row['split_date'])
        lgbm_model = lgbm_main(df=train_data.copy(), symbol=symbol, split_date=row['split_date'])
        stack_model = stacking_main(
            catboost_all=cat_model,
            xgb_all=xgb_model,
            lgbm_all=lgbm_model,
            df=train_data.copy(),
            symbol=symbol,
            split_date=row['split_date'],
            model_name=row['model_name']
        )
        file_pth = Path(f"model_outputs") / row['model_name'].lower().replace("{symbol}", symbol)
        s3_pkl_pth = f'models/{file_pth.name}'
        s3.upload_file(str(file_pth), bucket_name, s3_pkl_pth)
        del train_data

    # 6) (Optional) Backtest
    if row.get('test_model', 0) in [1, 0]:
        print(f"[{symbol}] back testing…")
        Path('backtest_results').mkdir(exist_ok=True)
        parquet_pth = f'parquet_files/{symbol}'
        test_df = load_data_backtest(
            parquet_pth,
            test_start_date=pd.to_datetime(row['test_start_date']),
            strike_diff=row["strike_diff"],
            le_days_to_expiry=row['le_days_to_expiry'],
            gt_days_to_expiry=row['gt_days_to_expiry'],
        ).sort_values('date')
        model_filename = Path("model_outputs") / row['model_name'].lower().replace("{symbol}", symbol)
        xl_file_pth = bt_main(
            df=test_df,
            model=joblib.load(model_filename),
            lot_quantity=row['lot_quantity'],
            symbol=symbol,
            strike_diff=row['strike_diff'],
            stop_loss_percent=row['stop_loss_percent'],
            target_percent=row['target_percent'],
            breakeven_pct=row['breakeven_pct'],
            breakeven_trail_pct=row['breakeven_trail_pct'],
            trail_trigger_pct=row['trail_trigger_pct'],
            trail_buffer_pct=row['trail_buffer_pct'],
            ce_threshold_min=row['ce_threshold_min'],
            pe_threshold_min=row['pe_threshold_min'],
            cpu_cores=min(cpu_count() - 1, 20)
        )
        s3_xl_pth = f'model-c/{xl_file_pth.name}'
        s3.upload_file(str(xl_file_pth), bucket_name, s3_xl_pth)
    print(f"[{symbol}] done.")
    return symbol


def main():
    total_start = datetime.now()
    print("Started:", total_start)

    # load your config
    config = pd.read_csv(
        'instrument_config_updated - Copy.csv',
        parse_dates=[
            'train_start_date',
            'cutoff_date',
            'split_date',
            'test_start_date'
        ]
    )
    global_start = config['train_start_date'].min()
    rows = [
        {**r, 'global_start': global_start}
        for r in config[config['active'].isin([1,0])].to_dict('records')
    ]

    print("One-time cache…")
    load_or_download_all_ohlc_parquet(
        bucket="analogyx-trading-data",
        prefix="NFO/",
        start_date=global_start,
        cpu_cores=min(cpu_count() - 1, 40)
    )

    n_symbols = len(rows)
    if n_symbols == 0:
        print("No active symbols to process.")
        return

    n_workers = min(1, n_symbols)
    print(f"Spawning {n_workers} threads for {n_symbols} symbols…")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_symbol, row): row['symbol'] for row in rows}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                fut.result()
                print(f"[{sym}] completed successfully.")
            except Exception as e:
                print(f"[{sym}] raised an exception: {e!r}")

    print("All symbols done.")
    print("Total time:", datetime.now() - total_start)


if __name__ == "__main__":
    main()
