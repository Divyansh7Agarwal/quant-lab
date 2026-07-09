"""Position sizing & risk management — the part that matters most when it's your
own money. Converts raw conviction signals into actual position weights with
volatility targeting and exposure caps."""
import numpy as np
import pandas as pd


def realized_vol(close, win=20, ppy=252):
    """Annualized rolling volatility of daily returns."""
    ret = close.pct_change()
    return ret.rolling(win).std() * np.sqrt(ppy)


def vol_target_position(signal, close, target_vol=0.15, ppy=252,
                        max_leverage=1.0, allow_short=False):
    """Scale a conviction signal [-1,1] into a position weight so that, at full
    conviction, the asset contributes ~`target_vol` annualized risk. Capped at
    `max_leverage`. This is the disciplined version of what was *accidentally*
    protecting Kronos — it cuts size when the asset gets choppy."""
    vol = realized_vol(close, ppy=ppy).replace(0, np.nan)
    scale = (target_vol / vol).clip(upper=max_leverage)
    pos = signal * scale
    if not allow_short:
        pos = pos.clip(lower=0.0)
    return pos.clip(-max_leverage, max_leverage).fillna(0.0)


def combine_portfolio(positions: dict, max_gross=1.0):
    """positions: {ticker -> Series of per-asset weights}. Aligns on a common
    calendar, then scales down any day whose gross exposure exceeds `max_gross`
    (no leverage by default — you never wake up owing money)."""
    df = pd.DataFrame(positions).sort_index()
    df = df.ffill().fillna(0.0)
    gross = df.abs().sum(axis=1)
    scale = (max_gross / gross).clip(upper=1.0).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return df.mul(scale, axis=0)


def drawdown_brake(equity, weights, dd_limit=0.20, cut=0.5):
    """Optional circuit breaker: if portfolio drawdown breaches `dd_limit`,
    cut exposure by `cut` until a new high is recovered. Crude but effective at
    surviving regime breaks."""
    eq = np.asarray(equity, float)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    brake = np.where(dd < -dd_limit, cut, 1.0)
    return weights.mul(pd.Series(brake, index=weights.index), axis=0)
