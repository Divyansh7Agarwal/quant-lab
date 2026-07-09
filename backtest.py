"""Walk-forward backtest engine. No look-ahead: positions decided at the close of
day t are held during day t+1 (enforced by .shift(1)). Costs charged on turnover."""
import numpy as np
import pandas as pd

import signals as sg
import portfolio as pf
import metrics as mx


def _align_returns(universe):
    """Union calendar of daily simple returns across the universe (0 on non-trading
    days for a given asset, so it just doesn't contribute that day)."""
    rets = {t: df["close"].pct_change() for t, df in universe.items()}
    R = pd.DataFrame(rets).sort_index()
    return R


def run_backtest(universe, cfg):
    closes = {t: df["close"] for t, df in universe.items()}

    # 1. per-asset conviction -> vol-targeted position
    positions = {}
    for t, close in closes.items():
        sig = sg.ensemble(close, cfg.get("signal_weights"),
                          mom_lookbacks=cfg.get("mom_lookbacks", (20, 60, 120)),
                          xover=cfg.get("xover", (20, 100)))
        positions[t] = pf.vol_target_position(
            sig, close,
            target_vol=cfg["per_asset_vol"],
            ppy=cfg["ppy"],
            max_leverage=cfg["max_leverage"],
            allow_short=cfg["allow_short"],
        )

    # 2. combine into a portfolio with a gross-exposure cap
    weights = pf.combine_portfolio(positions, max_gross=cfg["max_gross"])

    # 3. returns aligned to the weight calendar
    R = _align_returns(universe).reindex(weights.index).fillna(0.0)

    # 4. trade NEXT bar -> shift weights; charge cost on turnover
    held = weights.shift(1).fillna(0.0)
    turnover = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * (cfg["fee_bps"] / 1e4)

    gross_ret = (held * R).sum(axis=1)
    net_ret = gross_ret - cost
    exposure = held.abs().sum(axis=1)

    # 5. drop warmup (rolling windows not yet valid) and optional date floor
    start = weights.index[cfg["warmup"]] if len(weights) > cfg["warmup"] else weights.index[0]
    if cfg.get("start_date"):
        start = max(pd.Timestamp(cfg["start_date"]), start)
    mask = net_ret.index >= start
    net_ret, gross_ret, exposure, turnover = net_ret[mask], gross_ret[mask], exposure[mask], turnover[mask]

    # 6. equal-weight buy & hold of the same universe = the bar to beat
    bench = R[mask].mean(axis=1)

    res = {
        "net_ret": net_ret,
        "exposure": exposure,
        "turnover": turnover,
        "weights": weights[mask],
        "equity": (1 + net_ret).cumprod(),
        "bench_equity": (1 + bench).cumprod(),
        "strategy": mx.summary(net_ret, cfg["ppy"], exposure, turnover),
        "benchmark": mx.summary(bench, cfg["ppy"]),
        "period": (net_ret.index[0], net_ret.index[-1]) if len(net_ret) else (None, None),
        "n_days": int(len(net_ret)),
    }
    return res
