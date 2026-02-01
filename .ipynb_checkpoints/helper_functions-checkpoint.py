import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import pandas as pd
import requests

DB_DIR = Path("dbs")
BAD_SUFFIXES = ("UPUSDT","DOWNUSDT","BULLUSDT","BEARUSDT","3LUSDT","3SUSDT","5LUSDT","5SUSDT")


def interval_slug(s: str) -> str:
    return s.strip().replace(" ", "").replace("/", "").lower()

def make_db_name(pair: str, interval: str, weeks: int) -> str:
    return f"{pair}_{interval_slug(interval)}_{weeks}weeks.db"

def list_downloaded_pairs(db_dir=DB_DIR):
    """
    Returns list of symbols inferred from SQLite filenames in db_dir.
    Assumes file contains a token like BTCUSDT somewhere in the name.
    """
    symbols = set()

    for p in Path(db_dir).glob("*.db"):
        name = p.stem  # filename without extension
        # find the first token that looks like a Binance symbol ending with USDT
        m = re.search(r"([A-Z0-9]{2,20}USDT)", name.upper())
        if m:
            symbols.add(m.group(1))

    # filter out leveraged tokens etc
    symbols = [s for s in symbols if s.endswith("USDT") and not s.endswith(BAD_SUFFIXES)]
    return sorted(symbols)
    
def load_or_fetch_pair_df(pair: str, interval: str, weeks: int) -> tuple[str, "pd.DataFrame"]:
    db_name = make_db_name(pair, interval, weeks)
    db_path = "./db/" + db_name

    print(f"[{pair}] DB: {db_path}")

    binance_client = BinanceClient(db_path)
    binance_client.set_interval(interval)

    df = None

    if os.path.exists(db_path):
        df = binance_client.fetch_data_from_db(pair)
        if df is not None and not df.empty:
            print(f"[{pair}] Loaded {len(df):,} rows from DB.")
        else:
            df = None

    if df is None:
        print(f"[{pair}] No usable DB data found -> fetching from Binance...")

        api_secret = os.getenv("BINANCE_SECRET_KEY")
        api_key = os.getenv("BINANCE_API_KEY")
        binance_client.make(api_key, api_secret)

        server_time = binance_client.get_server_time()
        end_dt = datetime.fromtimestamp(server_time["serverTime"] / 1000, tz=timezone.utc)
        start_dt = end_dt - timedelta(weeks=weeks)

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        data = binance_client.fetch_data(pair, start_ms, end_ms)
        if data is None or data.empty:
            raise RuntimeError(f"[{pair}] No data returned from Binance for the requested range.")

        binance_client.store_data_to_db(pair, data)

        df = binance_client.fetch_data_from_db(pair)
        if df is None or df.empty:
            raise RuntimeError(f"[{pair}] Data fetched/stored but DB load returned empty.")

        print(f"[{pair}] Fetched + stored + loaded {len(df):,} rows.")

    df = df.sort_index()
    return db_path, df
