"""Crypto data layer — daily OHLCV for a universe of liquid coins via ccxt (Bybit).
Free public data, no API key. Caches per (symbol, timeframe), refreshes once/day.
Mirrors data.py so the backtest engine plugs in unchanged."""
import os, time
import datetime as _dt
import pandas as pd
import ccxt

CACHE = os.path.join(os.path.dirname(__file__), "cache", "crypto")
os.makedirs(CACHE, exist_ok=True)
_EX = ccxt.bybit({"enableRateLimit": True, "timeout": 15000})

# liquid USDT spot pairs with multi-year history (the reachable, tradeable alt universe)
UNIVERSE = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
            "LTC", "BCH", "ATOM", "ETC", "XLM", "NEAR", "FIL", "APT", "ARB", "OP",
            "INJ", "SUI", "SEI", "TIA", "AAVE", "UNI", "ALGO", "ICP"]


def _today():
    return _dt.date.today().isoformat()


def get(coin, timeframe="1d", days=1200, refresh=False):
    """Return DataFrame[open,high,low,close,volume] indexed by UTC timestamp."""
    sym = f"{coin}/USDT"
    path = os.path.join(CACHE, f"{coin}_{timeframe}.csv")
    if not refresh and os.path.exists(path):
        meta = path + ".date"
        if os.path.exists(meta) and open(meta).read().strip() == _today():
            return pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")

    ms_per = _EX.parse_timeframe(timeframe) * 1000
    since = _EX.milliseconds() - days * ms_per
    rows, cursor = [], since
    while True:
        batch = _EX.fetch_ohlcv(sym, timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows += batch
        cursor = batch[-1][0] + ms_per
        if len(batch) < 1000 or cursor >= _EX.milliseconds():
            break
        time.sleep(_EX.rateLimit / 1000)
    if not rows:
        raise RuntimeError(f"no data for {sym}")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts")
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
    df.reset_index().to_csv(path, index=False)
    open(path + ".date", "w").write(_today())
    return df


def get_universe(coins=None, min_days=400, **kw):
    """Dict of coin -> DataFrame, skipping coins with too little history."""
    out = {}
    for c in (coins or UNIVERSE):
        try:
            df = get(c, **kw)
            if len(df) >= min_days:
                out[c] = df
            else:
                print(f"  ~ {c}: only {len(df)}d history, skipped")
        except Exception as e:                       # noqa: BLE001
            print(f"  ! skip {c}: {e}")
    return out


def close_matrix(universe):
    """Align all coins onto one daily calendar → DataFrame of close prices."""
    closes = {c: df["close"] for c, df in universe.items()}
    return pd.DataFrame(closes).sort_index()


if __name__ == "__main__":
    u = get_universe()
    m = close_matrix(u)
    print(f"\n{len(u)} coins, {len(m)} days, {m.index[0].date()} → {m.index[-1].date()}")
