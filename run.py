"""Entry point. Edit CONFIG, run `python3 run.py`. Prints a metrics report and
saves an equity-curve PNG. This is the chassis a live system bolts onto."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data
from backtest import run_backtest

# ------------------------------- CONFIG -------------------------------
CONFIG = {
    # diversified, liquid universe — trend works best across many uncorrelated bets
    "universe": ["SPY", "QQQ", "GLD", "TLT", "EFA", "BTC-USD", "ETH-USD",
                 "AAPL", "MSFT", "NVDA", "AMZN"],
    "history": "6y",          # how much data to pull
    "start_date": None,       # e.g. "2021-01-01" to evaluate a sub-period

    # risk controls (the knobs that matter for real money)
    "per_asset_vol": 0.12,    # each asset sized to ~12% annualized vol at full conviction
    "max_leverage": 1.0,      # cap per-asset weight (1.0 = no leverage)
    "max_gross": 1.0,         # cap total exposure (1.0 = never more than 100% invested)
    "allow_short": False,     # long/flat only — safest for own capital
    "fee_bps": 10,            # per-trade cost incl. slippage, one-way (0.10%)

    "ppy": 252,               # annualization (252 trading days)
    "warmup": 120,            # drop first N days (rolling signals not yet valid)
    "signal_weights": {"tsmom": 0.6, "xover": 0.25, "meanrev": 0.15},
}
OUTDIR = os.path.dirname(__file__)
# ----------------------------------------------------------------------


def _fmt(d):
    pct = {"total_return", "cagr", "ann_vol", "max_dd", "hit_rate", "avg_exposure"}
    order = ["total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_dd",
             "calmar", "hit_rate", "profit_factor", "avg_exposure", "ann_turnover"]
    out = {}
    for k in order:
        if k not in d:
            continue
        v = d[k]
        if v == float("inf"):
            out[k] = "inf"
        elif k in pct:
            out[k] = f"{v*100:.1f}%"
        else:
            out[k] = f"{v:.2f}"
    return out


def main():
    print(f"Downloading {len(CONFIG['universe'])} assets...")
    uni = data.get_universe(CONFIG["universe"], period=CONFIG["history"])
    print(f"Loaded {len(uni)} assets: {', '.join(uni)}")

    res = run_backtest(uni, CONFIG)
    p0, p1 = res["period"]
    print(f"\nBacktest window: {p0.date()} -> {p1.date()}  ({res['n_days']} days)")

    s, b = _fmt(res["strategy"]), _fmt(res["benchmark"])
    keys = list(s.keys())
    print(f"\n  {'metric':<16}{'STRATEGY':>12}{'EW Buy&Hold':>14}")
    print("  " + "-" * 42)
    for k in keys:
        print(f"  {k:<16}{s.get(k,''):>12}{b.get(k,''):>14}")

    # equity curve
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(res["equity"].index, res["equity"].values, color="#16a34a", lw=1.6,
            label=f"Strategy (Sharpe {res['strategy']['sharpe']:.2f})")
    ax.plot(res["bench_equity"].index, res["bench_equity"].values, color="#6b7280",
            lw=1.3, ls="--", label=f"Equal-weight Buy&Hold (Sharpe {res['benchmark']['sharpe']:.2f})")
    ax.set_yscale("log")
    ax.set_title("Systematic trend portfolio vs buy & hold (net of costs)")
    ax.set_ylabel("growth of $1 (log)"); ax.legend(); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = os.path.join(OUTDIR, "equity_curve.png")
    plt.savefig(out, dpi=130); plt.close()
    print(f"\nSaved equity curve -> {out}")
    print("\nNext: prove an LLM news/sentiment signal raises Sharpe ON TOP of this,")
    print("then paper-trade before a rupee/dollar of real capital goes in.")


if __name__ == "__main__":
    main()
