"""Performance metrics. All functions take a pandas Series of *periodic* (e.g. daily)
strategy returns and an annualization factor (252 stocks, 365 crypto)."""
import numpy as np
import pandas as pd


def _arr(r):
    return np.asarray(r, dtype=float)


def cagr(ret, ppy):
    r = _arr(ret)
    if len(r) == 0:
        return 0.0
    growth = np.prod(1 + r)
    years = len(r) / ppy
    return float(growth ** (1 / years) - 1) if years > 0 and growth > 0 else float("nan")


def ann_vol(ret, ppy):
    return float(_arr(ret).std() * np.sqrt(ppy))


def sharpe(ret, ppy, rf=0.0):
    r = _arr(ret) - rf / ppy
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0


def sortino(ret, ppy, rf=0.0):
    r = _arr(ret) - rf / ppy
    downside = r[r < 0]
    dd = downside.std()
    return float(r.mean() / dd * np.sqrt(ppy)) if dd > 0 else 0.0


def max_drawdown(ret):
    eq = np.cumprod(1 + _arr(ret))
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min()) if len(eq) else 0.0


def calmar(ret, ppy):
    mdd = abs(max_drawdown(ret))
    return float(cagr(ret, ppy) / mdd) if mdd > 0 else float("inf")


def hit_rate(ret):
    r = _arr(ret)
    traded = r[r != 0]
    return float((traded > 0).mean()) if len(traded) else 0.0


def profit_factor(ret):
    r = _arr(ret)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else float("inf")


def total_return(ret):
    return float(np.prod(1 + _arr(ret)) - 1)


def summary(ret, ppy, exposure=None, turnover=None):
    """Return an ordered dict of headline metrics for a return series."""
    d = {
        "total_return": total_return(ret),
        "cagr": cagr(ret, ppy),
        "ann_vol": ann_vol(ret, ppy),
        "sharpe": sharpe(ret, ppy),
        "sortino": sortino(ret, ppy),
        "max_dd": max_drawdown(ret),
        "calmar": calmar(ret, ppy),
        "hit_rate": hit_rate(ret),
        "profit_factor": profit_factor(ret),
    }
    if exposure is not None:
        d["avg_exposure"] = float(np.mean(np.abs(_arr(exposure))))
    if turnover is not None:
        d["ann_turnover"] = float(np.sum(_arr(turnover)) / (len(turnover) / ppy))
    return d
