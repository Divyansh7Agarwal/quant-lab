"""Data layer: download daily OHLCV from Yahoo, cache to disk, reuse same-day.
Swap this out for a broker/exchange feed when going live — nothing else changes."""
import os
import datetime as _dt
import pandas as pd
import yfinance as yf

CACHE = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE, exist_ok=True)
COLS = ["open", "high", "low", "close", "volume"]


def _today():
    # Date.now-style call is fine in normal Python; only the workflow sandbox forbids it.
    return _dt.date.today().isoformat()


def get(ticker, period="6y", interval="1d", refresh=False):
    """Return a tidy DataFrame indexed by timestamp with open/high/low/close/volume.
    Caches per (ticker, interval) and only re-downloads once per day."""
    path = os.path.join(CACHE, f"{ticker.replace('/', '_')}_{interval}.csv")
    if not refresh and os.path.exists(path):
        meta = os.path.join(path + ".date")
        if os.path.exists(meta) and open(meta).read().strip() == _today():
            df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
            return df

    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"no data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    df = raw[COLS].dropna()
    df.index.name = "timestamp"
    df.reset_index().to_csv(path, index=False)
    with open(path + ".date", "w") as fh:
        fh.write(_today())
    return df


def get_universe(tickers, **kw):
    """Dict of ticker -> DataFrame, skipping any that fail to download."""
    out = {}
    for t in tickers:
        try:
            out[t] = get(t, **kw)
        except Exception as e:           # noqa: BLE001
            print(f"  ! skip {t}: {e}")
    return out
