"""Cross-sectional crypto momentum backtest — the first real edge test.

Each week: rank the alt universe by trailing return, go long the strongest K and
(optionally) short the weakest K, equal-weight. This isolates the momentum
ANOMALY (does relative strength persist?) from crypto's beta. Pure price data →
no leakage → an honest yes/no.

Reports one principled config in full, plus a robustness grid across lookbacks
and long-only vs long/short — so a single lucky setting can't masquerade as edge.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import crypto_data as cd

PPY = 365          # crypto trades every day
FEE_BPS = 10       # per-side cost on turnover (taker + slippage), basis points


# ------------------------------- metrics -------------------------------
def stats(daily):
    r = np.asarray(daily, float)
    r = r[~np.isnan(r)]
    if len(r) == 0 or r.std() == 0:
        return dict(ret=0, cagr=0, vol=0, sharpe=0, maxdd=0, hit=0)
    eq = np.cumprod(1 + r)
    yrs = len(r) / PPY
    peak = np.maximum.accumulate(eq)
    return dict(
        ret=eq[-1] - 1,
        cagr=eq[-1] ** (1 / yrs) - 1,
        vol=r.std() * np.sqrt(PPY),
        sharpe=r.mean() / r.std() * np.sqrt(PPY),
        maxdd=(eq / peak - 1).min(),
        hit=(r > 0).mean(),
    )


# ---------------------------- the strategy -----------------------------
def backtest(close, lookback=30, rebal=7, k=5, long_short=True, fee_bps=FEE_BPS):
    rets = close.pct_change()
    dates = close.index
    W = pd.DataFrame(0.0, index=dates, columns=close.columns)

    rebal_days = range(lookback + 1, len(dates), rebal)
    for i in rebal_days:
        t = dates[i]
        past = close.iloc[i - lookback]
        now = close.iloc[i]
        mom = (now / past - 1).dropna()
        mom = mom[np.isfinite(mom)]
        if len(mom) < 2 * k:
            continue
        ranked = mom.sort_values()
        w = pd.Series(0.0, index=close.columns)
        w[ranked.index[-k:]] = 1.0 / k                     # long strongest k
        if long_short:
            w[ranked.index[:k]] = -1.0 / k                 # short weakest k
        W.iloc[i] = w.values
    W = W.replace(0.0, np.nan).ffill().fillna(0.0)         # hold weights between rebalances

    gross = (W.shift(1) * rets).sum(axis=1)                # yesterday's book earns today's move
    turnover = W.diff().abs().sum(axis=1)
    net = gross - turnover * (fee_bps / 1e4)
    return net, turnover.sum() / (len(dates) / PPY)        # (daily returns, annualized turnover)


def fmt(s):
    return (f"{s['ret']*100:>8.0f}%{s['cagr']*100:>8.1f}%{s['vol']*100:>7.0f}%"
            f"{s['sharpe']:>7.2f}{s['maxdd']*100:>8.0f}%{s['hit']*100:>6.0f}%")


def main():
    print("Loading crypto universe...")
    uni = cd.get_universe()
    close = cd.close_matrix(uni)
    p0, p1 = close.index[0].date(), close.index[-1].date()
    print(f"{close.shape[1]} coins, {close.shape[0]} days ({p0} → {p1})\n")

    btc = close["BTC"].pct_change()
    eqw = close.pct_change().mean(axis=1)                  # equal-weight the whole universe

    # ---- primary config, full detail ----
    net, turn = backtest(close, lookback=30, rebal=7, k=5, long_short=True)
    hdr = f"  {'strategy':<22}{'return':>9}{'cagr':>8}{'vol':>7}{'sharpe':>7}{'maxDD':>8}{'hit':>6}"
    print("PRIMARY: 30d lookback, weekly rebalance, long top-5 / short bottom-5, net of 10bps")
    print(hdr); print("  " + "-" * 67)
    print(f"  {'XS-momentum L/S':<22}{fmt(stats(net))}   (turnover {turn:.0f}x/yr)")
    print(f"  {'BTC buy & hold':<22}{fmt(stats(btc))}")
    print(f"  {'equal-weight universe':<22}{fmt(stats(eqw))}")

    # ---- robustness grid (report ALL, no cherry-picking) ----
    print("\nROBUSTNESS — Sharpe across lookbacks × style (weekly rebal, k=5, net):")
    print(f"  {'lookback':<10}{'long/short':>12}{'long-only':>12}")
    print("  " + "-" * 34)
    for lb in (15, 30, 60, 90):
        ls = stats(backtest(close, lookback=lb, k=5, long_short=True)[0])["sharpe"]
        lo = stats(backtest(close, lookback=lb, k=5, long_short=False)[0])["sharpe"]
        print(f"  {lb:>4}d     {ls:>12.2f}{lo:>12.2f}")

    # ---- equity plot for the primary config ----
    fig, ax = plt.subplots(figsize=(11, 5))
    for series, lab, col in [(net, "XS-momentum L/S", "#7c3aed"),
                             (btc, "BTC buy & hold", "#6b7280"),
                             (eqw, "equal-weight universe", "#9ca3af")]:
        eq = (1 + series.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, label=lab, lw=1.5,
                color=col, ls="-" if lab.startswith("XS") else "--")
    ax.set_yscale("log")
    ax.set_title("Crypto cross-sectional momentum (30d, weekly, L/S top-5) vs benchmarks")
    ax.set_ylabel("growth of $1 (log)"); ax.legend(); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "crypto_momentum.png")
    plt.savefig(out, dpi=130); plt.close()
    print(f"\nSaved {out}")
    print("\nRead it honestly: is L/S momentum's Sharpe > ~1 AND consistent across lookbacks,")
    print("AND does it beat just holding BTC? If not, this edge isn't here either.")


if __name__ == "__main__":
    main()
