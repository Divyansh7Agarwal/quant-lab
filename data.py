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
    # UTC like the rest of the pipeline — Mac (IST) and cloud runner must agree.
    return _dt.datetime.now(_dt.timezone.utc).date().isoformat()


class SanityError(RuntimeError):
    """The feed returned data too stale or too absurd to act on."""


def last_close(ticker, max_stale_days=5, max_move=0.20):
    """Latest close, refused if it can't be trusted.

    Guards against the free feed's known failure modes: a series that quietly
    stopped updating days ago, or a bad tick (zero/negative/NaN, or a one-day
    move so large it's almost surely a data error, not a market). Callers skip
    the symbol for the day instead of trading/marking on garbage.
    """
    df = get(ticker, period="1y")
    c = df["close"].dropna()
    if len(c) < 2:
        raise SanityError(f"{ticker}: not enough history to sanity-check")
    last, prev = float(c.iloc[-1]), float(c.iloc[-2])
    age_days = (_dt.datetime.now(_dt.timezone.utc).date()
                - c.index[-1].date()).days
    if age_days > max_stale_days:
        raise SanityError(f"{ticker}: last price is {age_days} days old (feed stale)")
    if not last > 0:
        raise SanityError(f"{ticker}: non-positive close {last}")
    if prev > 0 and abs(last / prev - 1) > max_move:
        raise SanityError(f"{ticker}: {abs(last/prev-1)*100:.0f}% one-day jump "
                          f"({prev:g} → {last:g}) — likely bad data")
    return last


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
