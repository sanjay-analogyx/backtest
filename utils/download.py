import glob
import os
from concurrent.futures.process import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
import boto3
import pandas as pd
import pyarrow.parquet as pq
import io
import re
import concurrent
from multiprocessing import cpu_count
from datetime import datetime
from typing import Optional, Tuple
import numpy as np


def download_symbol_ohlc_from_s3(symbol: str, s3_bucket: str, s3_file_list: list):
    """
    Downloads and concatenates OHLC data for a specific symbol from a provided list of parquet files in an S3 bucket, without saving locally.

    Args:
        symbol (str): The symbol to download data for.
        s3_bucket (str): The S3 bucket name.
        s3_file_list (List[str]): List of S3 parquet file keys.

    Returns:
        pd.DataFrame: DataFrame containing concatenated OHLC data for the symbol.
    """
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    dfs = []
    selected_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'symbol', 'expiry_date',
                        'strike', 'option_type']
    for key in s3_file_list:
        try:
            key = f'GDF DATA/{key}'  # Ensure the key is prefixed correctly
            s3_obj = s3.get_object(Bucket=s3_bucket, Key=key)
            data = s3_obj['Body'].read()
            table = pq.read_table(io.BytesIO(data))
            df: pd.DataFrame = table.to_pandas()
            if pd.api.types.is_datetime64tz_dtype(df['date']):
                df['date'] = df['date'].dt.tz_localize(None)
            # Filter rows for the given symbol if symbol is a columns
            df.columns = df.columns.str.lower()  # Normalize column names to lowercase
            columns = {
                'openinterest': 'open_interest',
                'tradesymbol': 'symbol',
                'expiry': 'expiry_date',
                'tradedqty': 'volume',
                'strikeprice': 'strike',
                'optiontype': 'option_type'
            }
            df.rename(columns=columns, inplace=True)
            df['expiry_date'] = pd.to_datetime(df['expiry_date'], errors='coerce')
            df = df[selected_columns]
            df = df[df['symbol'].str.startswith(symbol, na=False)]  # Filter by symbol
            dfs.append(df)
        except Exception as e:
            print(f"Error processing {key}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=selected_columns)


# make sure AWS_KEYS are set somewhere above
AWS_ARGS = {
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY
}


def download_vix_from_s3(s3_bucket: str, key = "GDF DATA/vix.csv") -> pd.DataFrame:
    """
    Downloads VIX data from S3 (key: 'GDF DATA/vix.csv') and returns it as a pandas DataFrame.
    Caches the result in 'data/vix.csv' locally and only re-downloads if that file is missing.
    """
    # 1) Ensure local cache folder
    cache_dir = "data/cache"
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, "vix.csv")

    # 2) If cached, just load and return
    if 'INDIA VIX' not in key:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path, parse_dates=["date"], infer_datetime_format=True)
            return df

    # 3) Otherwise pull from S3
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=s3_bucket, Key=key)
        raw = obj["Body"].read()
        df = pd.read_csv(io.BytesIO(raw))
        df = pd.read_csv('vix_data.csv')

        # normalize and rename
        df.columns = df.columns.str.lower()
        df = df.rename(columns={
            "open": "vix_open",
            "high": "vix_high",
            "low": "vix_low",
            "close": "vix_close"
        })

        # parse date and select cols
        # if pd.api.types.is_datetime64tz_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].dt.tz_localize(None)
        df = df[["date", "vix_open", "vix_high", "vix_low", "vix_close"]]

        # 4) cache to disk
        df.to_csv(local_path, index=False)
        return df

    except Exception as e:
        print(f"Error downloading VIX data from S3: {e}")
        return pd.DataFrame(columns=["date", "vix_open", "vix_high", "vix_low", "vix_close"])


def download_symbol_ohlc_from_s3_csv(symbol: str, s3_bucket: str, s3_folder: str, start_date):
    """
    Downloads and concatenates OHLC data for a specific symbol from all CSV files in a given S3 folder, using multithreading.

    Args:
        symbol (str): The symbol to download data for.
        s3_bucket (str): The S3 bucket name.
        s3_folder (str): The S3 folder (prefix) containing CSV files.

    Returns:
        pd.DataFrame: DataFrame containing concatenated OHLC data for the symbol.
    """
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    dfs = []
    selected_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'symbol', 'expiry_date',
                        'strike', 'option_type']
    prefix = f'{s3_folder}/' if not s3_folder.endswith('/') else s3_folder

    def process_key(key):
        try:
            s3_obj = s3.get_object(Bucket=s3_bucket, Key=key)
            data = s3_obj['Body'].read()
            df = pd.read_csv(io.BytesIO(data))
            df.columns = df.columns.str.lower()
            columns = {
                'ticker': 'symbol',
                'date': 'only_date',
                'open interest': 'open_interest',
                'tradesymbol': 'symbol',
                'expiry': 'expiry_date',
                'optiontype': 'option_type',
            }
            df.rename(columns=columns, inplace=True)
            df = df[df['symbol'].str.startswith(symbol, na=False)]  # Filter by symbol
            pattern = re.compile(r"^([A-Z0-9]+)(\d{2}[A-Z]{3}\d{2})(\d+(?:\.\d+)?)([A-Z]+)\.NFO$")
            extracted = df['symbol'].str.extract(pattern)
            if 'expiry_date' not in df.columns:
                df['expiry_date'] = extracted[1]
            if 'strike' not in df.columns:
                df['strike'] = extracted[2]
            if 'option_type' not in df.columns:
                df['option_type'] = extracted[3]
            df['date'] = pd.to_datetime(df['only_date'] + ' ' + df['time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            df['expiry_date'] = pd.to_datetime(df['expiry_date'], errors='coerce')
            df = df[df['symbol'].str.startswith(symbol, na=False)]
            available_columns = [col for col in selected_columns if col in df.columns]
            df = df[available_columns]
            df = df[df['date'] >= start_date]
            if df.empty:
                0
            return df
        except Exception as e:
            print(f"Error processing {key}: {e}")
            return None

    keys = []
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.lower().endswith('.csv'):
                    keys.append(key)
    except Exception as e:
        print(f"Error listing objects in {prefix}: {e}")
    cpu_cores = min(cpu_count() - 1, 40)
    if keys:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
            results = list(executor.map(process_key, keys))
        dfs = [df for df in results if df is not None]

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=selected_columns)


def download_all_symbol_ohlc_from_s3(s3_bucket: str, s3_folder: str, start_date, keys=[]):
    """
    Downloads and concatenates OHLC data for a specific symbol from all CSV files in a given S3 folder, using multithreading.

    Args:
        symbol (str): The symbol to download data for.
        s3_bucket (str): The S3 bucket name.
        s3_folder (str): The S3 folder (prefix) containing CSV files.

    Returns:
        pd.DataFrame: DataFrame containing concatenated OHLC data for the symbol.
    """
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    dfs = []
    selected_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'symbol', 'expiry_date',
                        'strike', 'option_type']
    prefix = f'{s3_folder}/' if not s3_folder.endswith('/') else s3_folder

    def process_key(key):
        try:
            s3_obj = s3.get_object(Bucket=s3_bucket, Key=key)
            data = s3_obj['Body'].read()
            df = pd.read_csv(io.BytesIO(data))
            df.columns = df.columns.str.lower()
            columns = {
                'ticker': 'symbol',
                'date': 'only_date',
                'open interest': 'open_interest',
                'tradesymbol': 'symbol',
                'expiry': 'expiry_date',
                'optiontype': 'option_type',
            }
            df.rename(columns=columns, inplace=True)
            pattern = re.compile(r"^([A-Z0-9]+)(\d{2}[A-Z]{3}\d{2})(\d+(?:\.\d+)?)([A-Z]+)\.NFO$")
            extracted = df['symbol'].str.extract(pattern)
            if 'expiry_date' not in df.columns:
                df['expiry_date'] = extracted[1]
            if 'strike' not in df.columns:
                df['strike'] = extracted[2]
            if 'option_type' not in df.columns:
                df['option_type'] = extracted[3]
            df['date'] = pd.to_datetime(df['only_date'] + ' ' + df['time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            df['expiry_date'] = pd.to_datetime(df['expiry_date'], errors='coerce')
            available_columns = [col for col in selected_columns if col in df.columns]
            df = df[available_columns]
            df = df[df['date'] >= start_date]
            if df.empty:
                0
            return df
        except Exception as e:
            print(f"Error processing {key}: {e}")
            return None

    keys = []
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.lower().endswith('.csv'):
                    keys.append(key)
    except Exception as e:
        print(f"Error listing objects in {prefix}: {e}")
    cpu_cores = min(cpu_count() - 1, 40)
    if keys:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
            results = list(executor.map(process_key, keys))
        dfs = [df for df in results if df is not None]

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=selected_columns)


AWS_ARGS = {
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY
}


def _process_and_save_file(args: Tuple[str, str, str, pd.Timestamp]):
    """
    Worker to download one CSV, process it, and write one Parquet.
    args = (s3_bucket, key, date_str, start_date, cache_dir)
    """
    s3_bucket, key, date_str, start_date, cache_dir = args
    s3 = boto3.client("s3", **AWS_ARGS)
    out_path = os.path.join(cache_dir, f"{date_str}.parquet")
    if os.path.exists(out_path):
        return

    obj = s3.get_object(Bucket=s3_bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    df.columns = df.columns.str.lower()
    df.rename(columns={
        "ticker": "symbol",
        "trade_symbol": "symbol",
        "open interest": "open_interest",
        "expiry": "expiry_date",
        "optiontype": "option_type",
    }, inplace=True)

    # fill missing expiry/strike/option_type
    # pattern = r"^([A-Z0-9-]+)(\d{2}[A-Z]{3}\d{2})(\d+(?:\.\d+)?)([CP]E)\.NFO$"
    # pat = re.compile(pattern)
    # .NFO is optional, so make it non-capturing and optional
    pat = re.compile(r"^([A-Z0-9]+)(\d{2}[A-Z]{3}\d{2})(\d+(?:\.\d+)?)([A-Z]+)(?:\.NFO)?$")
    # pat = re.compile(r"^([A-Z0-9]+)(\d{2}[A-Z]{3}\d{2})(\d+(?:\.\d+)?)([A-Z]+)\.NFO$")
    if "symbol" in df.columns:
        ext = df["symbol"].str.extract(pat)
        df["expiry_date"] = df.get("expiry_date", ext[1])
        df["strike"] = df.get("strike", ext[2])
        df["option_type"] = df.get("option_type", ext[3])
    try:
        df["date"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"],
            format="%d/%m/%Y %H:%M:%S"
        )
    except Exception as e:
        df["date"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"],
            format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    df = df[df["date"] >= start_date]

    keep = [
        "date", "open", "high", "low", "close", "volume",
        "open_interest", "symbol", "expiry_date", "strike", "option_type"
    ]
    df = df[[c for c in keep if c in df.columns]]
    df.to_parquet(out_path, index=False)


def cache_daily_ohlc(
        s3_bucket: str,
        s3_prefix: str,
        cache_dir: str,
        start_date: pd.Timestamp,
        cpu_cores
):
    """
    Splits the work of downloading/processing N daily CSVs
    across a process pool so you actually use all your cores.
    """
    os.makedirs(cache_dir, exist_ok=True)
    s3 = boto3.client("s3", **AWS_ARGS)

    # 1) list & filter keys
    date_rx = re.compile(r".*_(\d{8})\.csv$", re.IGNORECASE)
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=s3_bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            m = date_rx.match(key)
            if not m:
                continue
            dt = pd.to_datetime(m.group(1), format="%d%m%Y", errors="coerce")
            if pd.isna(dt) or dt < start_date:
                continue
            keys.append((key, m.group(1)))

    # 2) build argument tuples for each day
    tasks = [
        (s3_bucket, key, date_str, start_date, cache_dir)
        for key, date_str in keys
    ]

    # 3) parallelize!
    with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        # `.map` will block until all are done
        list(executor.map(_process_and_save_file, tasks))

    print(f"Cached {len(tasks)} days → {cache_dir}")


def _read_parquet_file(path: str, symbol_prefix) -> pd.DataFrame:
    if path == 'data/cache\\ohlc_nfo\\14072025.parquet':
        0
    df = pd.read_parquet(path)
    df = df[df["symbol"].str.startswith(symbol_prefix, na=False)]
    return df


def load_cached_ohlc(
        cache_dir: str,
        symbol: str,
        cpu_cores
) -> pd.DataFrame:
    # 1) find all the per-day parquet files
    files = glob.glob(os.path.join(cache_dir, "*.parquet"))
    if not files:
        return pd.DataFrame(columns=[
            "date", "open", "high", "low", "close", "volume",
            "open_interest", "symbol", "expiry_date", "strike", "option_type"
        ])

    # 2) read them in parallel across processes
    with ProcessPoolExecutor(max_workers=cpu_cores) as exe:
        # repeat(symbol_prefix) produces an infinite iterable of the same prefix
        dfs = list(exe.map(_read_parquet_file, files, repeat(symbol)))
    # 3) concat & optional filter
    df = pd.concat(dfs, ignore_index=True)

    return df.sort_values("date")


def load_or_download_all_ohlc_parquet(
        bucket: str,
        prefix: str,
        start_date: pd.Timestamp,
        cpu_cores,
        symbol: Optional[str] = None,
        cache_base: str = "data/cache"
) -> pd.DataFrame:
    cache_dir = os.path.join(cache_base, "ohlc_nfo")
    os.makedirs(cache_dir, exist_ok=True)

    # # build per-day Parquets if none exist
    # if not any(f.endswith(".parquet") for f in os.listdir(cache_dir)):
    #     print("Building One time Chache")
    #     cache_daily_ohlc(bucket, prefix, cache_dir, start_date, cpu_cores)

    # build per-day Parquets if none exist
    # if not any(f.endswith(".parquet") for f in os.listdir(cache_dir)):
    if symbol is None:
        print("Building One time Chache")
        cache_daily_ohlc(bucket, prefix, cache_dir, start_date, cpu_cores)

    df = None
    # If a symbol prefix is given, filter by startswith
    if symbol:
        df = load_cached_ohlc(cache_dir, symbol, cpu_cores)

    return df



def download_symbol_ohlc_from_s3_new(key):
    """
    Downloads and concatenates OHLC data for a specific symbol from a provided list of parquet files in an S3 bucket, without saving locally.

    Args:
        symbol (str): The symbol to download data for.
        s3_bucket (str): The S3 bucket name.
        s3_file_list (List[str]): List of S3 parquet file keys.

    Returns:
        pd.DataFrame: DataFrame containing concatenated OHLC data for the symbol.
    """
    # s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
    #                 aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    s3 = boto3.client('s3')
    selected_columns = [
        'date', 'open', 'high', 
        'low', 'close', 'volume', 
        'open_interest', 'symbol', 
        'expiry_date', 'strike', 'option_type'
        ]
    columns = {
        'openinterest': 'open_interest',
        'tradesymbol': 'symbol',
        'expiry': 'expiry_date',
        'tradedqty': 'volume',
        'strikeprice': 'strike',
        'optiontype': 'option_type',
        'ticker': 'symbol',
        'open interest': 'open_interest'
    }
    Path('GDF DATA').mkdir(exist_ok=True)
    try:
        key = f'GDF DATA/{key}'  # Ensure the key is prefixed correctly
        if Path(key).exists():
            return
        s3_obj = s3.get_object(Bucket='analogyx-trading-data', Key=key)
        data = s3_obj['Body'].read()
        table = pq.read_table(io.BytesIO(data))
        df: pd.DataFrame = table.to_pandas()
        # df = pd.read_parquet(key, engine='pyarrow')
        df.columns = df.columns.str.lower()
        df.rename(columns=columns, inplace=True)
        # if pd.api.types.is_datetime64tz_dtype(df['date']):
        if 'expiry_date' not in df.columns:
            df['symbol'] = df['symbol'].replace('\.\w+', '', regex=True)
            try:
                df['date'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'], format='%d/%m/%Y %H:%M:%S')
            except Exception as e:
                df['date'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S')
            # df['date'] = df['date'].dt.floor('min')
            
            # Try to extract option-style (with strike and option_type) and future-style (ending with FUT)
            pat_opt = re.compile(r"^([A-Z0-9-]+)(\d{2}[A-Z]{3}\d{2})(\d+(?:\.\d+)?)([A-Z]+)$")
            pat_fut = re.compile(r"^([A-Z0-9-]+)(\d{2}[A-Z]{3}\d{2})FUT$")
            
            # First try option pattern
            ext_opt = df["symbol"].str.extract(pat_opt)
            # Then try future pattern
            ext_fut = df["symbol"].str.extract(pat_fut)
            
            # If expiry_date not in columns, fill from whichever pattern matches
            expiry_col = df.get("expiry_date", ext_opt[1].combine_first(ext_fut[1]))
            
            # Accept expiry in DDMMMYY (e.g., 26JUN25)
            valid_expiry_mask = expiry_col.astype(str).str.match(r"^\d{2}[A-Z]{3}\d{2}$")
            expiry_col_valid = expiry_col.where(valid_expiry_mask, None)
            
            df["expiry_date"] = pd.to_datetime(expiry_col_valid, errors='coerce', format="%d%b%y")
            # For strike and option_type, use option pattern if available, else set as NaN or 'XX' for futures
            df["strike"] = df.get("strike", ext_opt[2])
            df["option_type"] = df.get("option_type", ext_opt[3].where(ext_opt[3].notna(), 'FUT'))
            df.loc[df["option_type"] == 'FUT', "strike"] = np.nan  # No strike for futures
            df['option_type'] = df['option_type'].replace('FUT', 'XX')
            # df['option_type'] = df['option_type'].replace(np.nan, 'XX')
        df['expiry_date'] = pd.to_datetime(df['expiry_date'], errors='coerce')
        if isinstance(df['date'], pd.DatetimeTZDtype):
            df['date'] = df['date'].dt.tz_localize(None)
        # Filter rows for the given symbol if symbol is a columns
        df = df[selected_columns]
        # df = df[df['symbol'].str.startswith(symbol, na=False)]  # Filter by symbol
        key = f'GDF DATA/{Path(key).name}'
        df.to_parquet(key)
        return df
    except Exception as e:
        # print(f"Error processing {key}: {e}")
        return


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


