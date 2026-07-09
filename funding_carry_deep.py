"""Funding carry, stress-tested through a real bear market.

Pulls long funding history from Binance (reaches the 2021 euphoria + 2022 collapse),
then asks the questions that decide whether this is fundable:
  1. How does the carry look across regimes (fat bull vs negative bear)?
  2. What does LEVERAGE actually do to the drawdown when funding goes negative?
  3. Does a simple regime rule (only hold while funding is positive) fix the bear bleed?

Objective free data → honest. What it still CAN'T show: liquidation cascades,
basis blowouts, and exchange insolvency — the real tail. Those we reason about, not backtest.
"""
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ccxt

_EX = ccxt.binance({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "future"}})
PPY_F = 3 * 365                     # funding paid every 8h
START = "2021-01-01"
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
SWITCH_BPS = 12                     # cost each time the timed version flips on/off


def fetch_funding(coin, start=START):
    sym = f"{coin}/USDT:USDT"
    since = _EX.parse8601(start + "T00:00:00Z")
    rows, cursor = [], since
    while True:
        batch = _EX.fetch_funding_rate_history(sym, since=cursor, limit=1000)
        if not batch:
            break
        rows += batch
        cursor = batch[-1]["timestamp"] + 1
        if len(batch) < 1000 or cursor >= _EX.milliseconds():
            break
        time.sleep(_EX.rateLimit / 1000)
    s = pd.Series({pd.to_datetime(r["timestamp"], unit="ms"): float(r["fundingRate"]) for r in rows})
    return s[~s.index.duplicated()].sort_index()


def metrics(ret):
    r = np.asarray(ret, float); r = r[~np.isnan(r)]
    if len(r) == 0 or r.std() == 0:
        return dict(ann=0, vol=0, sharpe=0, maxdd=0)
    eq = np.cumprod(1 + r); peak = np.maximum.accumulate(eq)
    return dict(ann=(1 + r.mean()) ** PPY_F - 1, vol=r.std() * np.sqrt(PPY_F),
                sharpe=r.mean() / r.std() * np.sqrt(PPY_F), maxdd=(eq / peak - 1).min())


def main():
    print(f"Fetching long funding history from Binance (since {START})...")
    funds = {}
    for c in COINS:
        try:
            s = fetch_funding(c)
            if len(s) > 500:
                funds[c] = s
                print(f"  {c:<5} {len(s):>5} pts  {s.index[0].date()} → {s.index[-1].date()}")
        except Exception as e:                       # noqa: BLE001
            print(f"  ! {c}: {str(e)[:60]}")

    F = pd.DataFrame(funds).sort_index()
    basket = F.mean(axis=1).dropna()                 # equal-weight carry, per 8h
    yrs = (basket.index[-1] - basket.index[0]).days / 365
    print(f"\n{F.shape[1]} coins, {len(basket)} periods, {yrs:.1f} years "
          f"({basket.index[0].date()} → {basket.index[-1].date()})\n")

    # ---- carry by calendar year: the regime story ----
    print("  annualized funding by year (equal-weight basket):")
    by_year = basket.groupby(basket.index.year).apply(lambda x: (1 + x.mean()) ** PPY_F - 1)
    for y, v in by_year.items():
        print(f"    {y}   {v*100:>7.1f}%")

    # ---- static vs regime-timed, unlevered and levered ----
    trail = basket.rolling(21).mean()                # ~7-day trailing funding (21 * 8h)
    on = (trail > 0).astype(float).shift(1).fillna(0.0)   # hold only when funding trending positive
    switch_cost = on.diff().abs() * (SWITCH_BPS / 1e4)
    timed = basket * on - switch_cost

    print("\n  strategy / leverage        ann.ret     vol   sharpe   maxDD")
    print("  " + "-" * 58)
    for lev in (1, 5, 10):
        st = metrics(basket * lev)
        tm = metrics(timed * lev)
        print(f"  static  {lev:>2}x               {st['ann']*100:>7.1f}%{st['vol']*100:>7.1f}%"
              f"{st['sharpe']:>8.2f}{st['maxdd']*100:>8.1f}%")
        print(f"  timed   {lev:>2}x (off if funding<0){tm['ann']*100:>6.1f}%{tm['vol']*100:>7.1f}%"
              f"{tm['sharpe']:>8.2f}{tm['maxdd']*100:>8.1f}%")

    # ---- plot: 5x static vs 5x timed across the full history ----
    fig, ax = plt.subplots(figsize=(11, 5))
    for series, lab, col in [((basket * 5), "5x static", "#dc2626"),
                             ((timed * 5), "5x timed (flat when funding<0)", "#16a34a")]:
        eq = (1 + series.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, label=lab, lw=1.5, color=col)
    ax.set_yscale("log")
    ax.set_title("Levered funding carry through 2021 euphoria + 2022 bear + recovery")
    ax.set_ylabel("growth of $1 (log)"); ax.legend(); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "funding_carry_deep.png")
    plt.savefig(out, dpi=130); plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
