import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from utils.filters import filter_option_data, filter_underlying_data
from utils.nifty_features_engineering_v2 import fe_main
from utils.common import create_date_series, get_last_thursday_of_month
from utils.download import download_symbol_ohlc_from_s3_new, download_vix_from_s3
from utils.backtest_model_parquet_stacking_v2 import bt_main
from multiprocessing import cpu_count
import joblib
from utils.filters import load_data_backtest
from pathlib import Path
from datetime import datetime, timedelta
from mpire import WorkerPool


class Backtest:
    def __init__(self, symbol):
        self.symbol = symbol
        self.s3_keys = self.get_file_list()


    def __call__(self, row):
        Path('backtest_results').mkdir(exist_ok=True)
        # if row['skip_fe']==1:
        if row['skip_fe'] in [1, 0]:
            current_year = datetime.now().year
            current_month = datetime.now().month
            last_thursday = get_last_thursday_of_month(current_year, current_month)
            last_day_of_month = (datetime(current_year, current_month, 1) + pd.offsets.MonthEnd(0)).date()
            today = datetime.now().date()
            if last_thursday.date() < today <= last_day_of_month:
                current_month += 1
                last_thursday = get_last_thursday_of_month(current_year, current_month)
            self.download_files()
            df = self.load_gdf_data()
            if self.symbol in ['NIFTY', 'BANKNIFTY']:
                upcoming_thursday = (datetime.now() + timedelta(days=(3 - datetime.now().weekday() + 7) % 7)).date()
                df = df[df['expiry_date'].dt.date==upcoming_thursday]
            else:
                df = df[df['expiry_date']==last_thursday]
            df = df[df['expiry_date']==last_thursday]
            df['strike'] = df['strike'].astype(float)
            df['date'] = df['date'].dt.floor('min')
            opt_data = df[df['option_type'].isin(['CE', 'PE'])]
            opt = filter_option_data(opt_data, days_to_expiry=64)
            fut_data = df[df['option_type']=='XX']
            fut = filter_underlying_data(fut_data)
            vix = download_vix_from_s3('analogyx-trading-data', "GDF DATA/INDIA VIX.csv")
            vix['date'] = vix['date'].dt.floor('min')
            fe_main(
                symbol=self.symbol,
                df_nifty=fut,
                df_option=opt,
                df_vix=vix,
                start_date=pd.to_datetime('today').date()-pd.Timedelta(days=15),
                strike_diff=row['strike_diff'],
                stop_loss_percent=row['stop_loss_percent'],
                target_percent=row['target_percent'],
                cpu_cores=min(cpu_count() - 1, 40)
            )
        parquet_pth = f'parquet_files/{self.symbol}'
        test_df = load_data_backtest(
            parquet_pth,
            test_start_date=pd.to_datetime(pd.to_datetime('today').date())-pd.Timedelta(days=15),
            strike_diff=row["strike_diff"],
            le_days_to_expiry=row['le_days_to_expiry'],
            gt_days_to_expiry=row['gt_days_to_expiry'],
        ).sort_values('date')
        model_filename = Path("model_outputs") / row['model_name'].lower().replace("{symbol}", self.symbol)
        bt_main(
            df=test_df,
            model=joblib.load(model_filename),
            lot_quantity=row['lot_quantity'],
            symbol=self.symbol,
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


    def get_file_list(self):
        dates = create_date_series(pd.to_datetime('today').date()-pd.Timedelta(days=15), pd.to_datetime('today').date())
        s3_keys = [f'{i}.parquet' for i in dates]
        return s3_keys


    def download_files(self):
        # files = Path('parq_for_backtest_july_8').glob(f'*.parquet')
        # for i in files:
        #     download_symbol_ohlc_from_s3_new(i)
        with WorkerPool() as exe:
            exe.map(download_symbol_ohlc_from_s3_new, self.s3_keys, progress_bar=True)
        return


    def load_file(self, pth):
            try:
                pth = f'GDF DATA/{pth}'
                df = pd.read_parquet(pth)
                df['symbol'] 
                return  df[df['symbol'].str.startswith(self.symbol, na=False)]
            except:
                return


    def load_gdf_data(self): 
        with WorkerPool(n_jobs=1) as exe:
            results = exe.map(self.load_file, self.s3_keys, progress_bar=True)
        return pd.concat(results, ignore_index=True)


def main(data):
    try:
        idx, row = data
        print(row['symbol'])
        obj = Backtest(row['symbol'])(row)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    symbol = 'BEL'
    config = pd.read_csv('instrument_config_updated.csv')
        # files = [i.name for i in Path('parquet_files').glob('*')]
    config = config[config['symbol']==symbol]
    # config = config[config['active']==1]
    for idx, row in config.iterrows():
        print(row['symbol'])
        # if row['symbol'] in files:
        #     continue
        # if Path(f'backtest_results/{row["symbol"]}').exists():
        #     continue
        obj = Backtest(row['symbol'])(row)
        # try:
        # except Exception as e:
        #     print(e)
    
    # with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
    #     executor.map(main, config.iterrows())
    #