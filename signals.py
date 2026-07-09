"""Signal generators. Each returns a pandas Series in roughly [-1, 1] aligned to the
price index: +1 = max long conviction, -1 = max short, 0 = flat.

CRITICAL: every signal at time t may only use information available up to and
including t. We trade on it the NEXT bar (handled in backtest.py via .shift(1)).
This is what keeps the backtest free of look-ahead bias.

This is the extension point: an LLM news/sentiment signal becomes just another
function here returning a Series, then gets a weight in `ensemble`.
"""
import numpy as np
import pandas as pd


def _z(series, win):
    m = series.rolling(win).mean()
    sd = series.rolling(win).std()
    return (series - m) / sd


def timeseries_momentum(close, lookbacks=(20, 60, 120)):
    """Multi-timeframe trend: average the sign of trailing returns over several
    horizons. More robust than a single lookback. Output in [-1, 1]."""
    sigs = []
    for lb in lookbacks:
        trailing = close / close.shift(lb) - 1
        sigs.append(np.tanh(trailing * 3))      # squashed, keeps magnitude info
    sig = pd.concat(sigs, axis=1).mean(axis=1)
    return sig.clip(-1, 1)


def ma_crossover(close, fast=20, slow=100):
    """Classic trend filter: +1 when fast MA above slow MA, else -1."""
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    return np.sign(f - s).fillna(0.0)


def mean_reversion(close, win=5, clip=2.0):
    """Short-horizon reversal: fade stretches from the rolling mean.
    Negative z-score (oversold) -> long. Small weight; it's a noisy edge."""
    z = _z(close, win)
    return (-z / clip).clip(-1, 1).fillna(0.0)


def ensemble(close, weights=None, mom_lookbacks=(20, 60, 120), xover=(20, 100)):
    """Blend signals into one conviction series in [-1, 1].
    Default leans on trend (what actually showed edge), dusts in reversal.
    Slower mom_lookbacks/xover -> fewer flips -> lower turnover (CTA style)."""
    weights = weights or {"tsmom": 0.6, "xover": 0.25, "meanrev": 0.15}
    parts = {
        "tsmom": timeseries_momentum(close, lookbacks=mom_lookbacks),
        "xover": ma_crossover(close, fast=xover[0], slow=xover[1]),
        "meanrev": mean_reversion(close),
    }
    sig = sum(weights[k] * parts[k] for k in weights)
    total_w = sum(weights.values())
    return (sig / total_w).clip(-1, 1).fillna(0.0)
