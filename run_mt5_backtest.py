"""Backtest the MT5 trend book on Mac, using Yahoo proxies for each MT5 symbol so
you get a real baseline on the SAME instruments you'll trade live. Runnable here;
the live loop (live_mt5.py) runs on the Windows VPS.

Trend-following on forex/metals/indices/commodities is the classic 'managed futures'
(CTA) playbook — exactly what this book is, and where trend historically works best.
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data
from backtest import run_backtest

# MT5 symbol  ->  Yahoo proxy (same underlying)
PROXY = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X",
    "XAUUSD": "GC=F", "XAGUSD": "SI=F", "USOIL": "CL=F",
    "US500": "ES=F", "USTEC": "NQ=F", "BTCUSD": "BTC-USD",
}

CONFIG = {
    "universe": list(PROXY.values()),
    "history": "6y", "start_date": None,
    "per_asset_vol": 0.10, "max_leverage": 0.5, "max_gross": 1.5,
    "allow_short": True,            # forex/CFD trend is naturally two-sided
    "fee_bps": 3,                   # FX/CFD spreads are tight on majors; ~3bps/side
    "ppy": 252, "warmup": 260,
    # SLOW-CTA config: long trend lookbacks, no fast mean-reversion -> low turnover
    "signal_weights": {"tsmom": 0.75, "xover": 0.25, "meanrev": 0.0},
    "mom_lookbacks": (60, 120, 250), "xover": (50, 200),
}


def _fmt(d):
    pct = {"total_return", "cagr", "ann_vol", "max_dd", "hit_rate", "avg_exposure"}
    order = ["total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_dd",
             "calmar", "hit_rate", "profit_factor", "avg_exposure", "ann_turnover"]
    out = {}
    for k in order:
        if k not in d:
            continue
        v = d[k]
        out[k] = "inf" if v == float("inf") else (f"{v*100:.1f}%" if k in pct else f"{v:.2f}")
    return out


def main():
    print(f"Downloading {len(CONFIG['universe'])} MT5-proxy instruments...")
    uni = data.get_universe(CONFIG["universe"], period=CONFIG["history"])
    rev = {v: k for k, v in PROXY.items()}
    print("Loaded:", ", ".join(f"{rev.get(t,t)}" for t in uni))

    res = run_backtest(uni, CONFIG)
    p0, p1 = res["period"]
    print(f"\nWindow: {p0.date()} -> {p1.date()}  ({res['n_days']} days), long/short, net of costs")

    s, b = _fmt(res["strategy"]), _fmt(res["benchmark"])
    print(f"\n  {'metric':<16}{'TREND BOOK':>13}{'EW Buy&Hold':>14}")
    print("  " + "-" * 43)
    for k in s:
        print(f"  {k:<16}{s.get(k,''):>13}{b.get(k,''):>14}")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(res["equity"].index, res["equity"].values, color="#16a34a", lw=1.6,
            label=f"MT5 trend book (Sharpe {res['strategy']['sharpe']:.2f})")
    ax.plot(res["bench_equity"].index, res["bench_equity"].values, color="#6b7280",
            lw=1.3, ls="--", label=f"Buy & hold (Sharpe {res['benchmark']['sharpe']:.2f})")
    ax.set_yscale("log")
    ax.set_title("MT5 trend book (forex/metals/indices/crypto) vs buy & hold")
    ax.set_ylabel("growth of $1 (log)"); ax.legend(); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "equity_mt5.png")
    plt.savefig(out, dpi=130); plt.close()
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
