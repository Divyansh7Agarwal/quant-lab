"""Funding-rate carry — the structural, market-neutral edge test.

On perpetual futures, leveraged longs pay a periodic 'funding' fee to shorts
(every 8h on Bybit). In bull/neutral regimes funding is persistently POSITIVE,
so: hold spot + short the perp (delta-neutral, no price bet) and collect funding.

This is 'get paid to bear a risk others want off their books', not a prediction.
Funding history is objective and free → honest backtest, no leakage. The question:
is the harvested yield real, steady, and big enough to beat costs?
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import ccxt

_EX = ccxt.bybit({"enableRateLimit": True, "timeout": 15000, "options": {"defaultType": "swap"}})
PPY = 365

# most liquid perps (deep books → the basis trade is actually executable)
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
PERIODS_PER_YEAR = 3 * 365          # funding paid every 8h
ROUNDTRIP_BPS = 12                  # enter spot+perp and exit: ~cost of the pair


def fetch_funding(coin, days=540):
    sym = f"{coin}/USDT:USDT"
    since = _EX.milliseconds() - days * 86400_000
    rows, cursor = [], since
    while True:
        batch = _EX.fetch_funding_rate_history(sym, since=cursor, limit=200)
        if not batch:
            break
        rows += batch
        cursor = batch[-1]["timestamp"] + 1
        if len(batch) < 200 or cursor >= _EX.milliseconds():
            break
        time.sleep(_EX.rateLimit / 1000)
    if not rows:
        raise RuntimeError(f"no funding for {sym}")
    s = pd.Series({pd.to_datetime(r["timestamp"], unit="ms"): float(r["fundingRate"]) for r in rows})
    return s[~s.index.duplicated()].sort_index()


def stats(per_period_ret):
    r = np.asarray(per_period_ret, float)
    r = r[~np.isnan(r)]
    if len(r) == 0 or r.std() == 0:
        return dict(ann=0, vol=0, sharpe=0, maxdd=0, posfrac=0)
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    return dict(
        ann=(1 + r.mean()) ** PERIODS_PER_YEAR - 1,        # annualized carry
        vol=r.std() * np.sqrt(PERIODS_PER_YEAR),
        sharpe=r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR),
        maxdd=(eq / peak - 1).min(),
        posfrac=(r > 0).mean(),
    )


def main():
    print("Fetching funding-rate history (Bybit perps)...")
    funds = {}
    for c in COINS:
        try:
            funds[c] = fetch_funding(c)
            print(f"  {c:<5} {len(funds[c])} funding points, "
                  f"{funds[c].index[0].date()} → {funds[c].index[-1].date()}")
        except Exception as e:                       # noqa: BLE001
            print(f"  ! {c}: {e}")

    F = pd.DataFrame(funds).sort_index()
    # long spot / short perp → the short perp RECEIVES funding when it's positive.
    # per-period return of the delta-neutral book = +funding (equal-weight basket).
    basket = F.mean(axis=1).dropna()

    # amortize round-trip cost across a typical ~30-day hold
    holds_per_year = PERIODS_PER_YEAR / (30 * 3)
    cost_drag = (ROUNDTRIP_BPS / 1e4) * holds_per_year

    print(f"\n{F.shape[1]} coins, {len(basket)} 8h periods "
          f"({basket.index[0].date()} → {basket.index[-1].date()})\n")

    s = stats(basket.values)
    print("Long-spot / short-perp carry basket (delta-neutral, equal-weight):")
    print(f"  gross annualized carry : {s['ann']*100:>6.1f}%")
    print(f"  est. cost drag         : {cost_drag*100:>6.1f}%/yr  (~monthly rebalance)")
    print(f"  NET annualized carry   : {(s['ann']-cost_drag)*100:>6.1f}%")
    print(f"  annualized vol         : {s['vol']*100:>6.1f}%")
    print(f"  Sharpe (gross)         : {s['sharpe']:>6.2f}")
    print(f"  max drawdown           : {s['maxdd']*100:>6.1f}%")
    print(f"  funding positive       : {s['posfrac']*100:>5.0f}% of periods")

    # per-coin average annualized funding, to see where the carry concentrates
    print("\n  per-coin avg annualized funding:")
    ann_by = ((1 + F.mean()) ** PERIODS_PER_YEAR - 1).sort_values(ascending=False)
    for c, v in ann_by.items():
        print(f"    {c:<5} {v*100:>6.1f}%")

    print("\n  Verdict test: is NET carry clearly positive with Sharpe > ~1 and a")
    print("  tolerable drawdown? A steady market-neutral yield IS a real edge —")
    print("  the first thing here that wouldn't just be betting on direction.")


if __name__ == "__main__":
    main()
