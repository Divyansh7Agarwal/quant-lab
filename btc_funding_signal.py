"""Funding extremes as a DIRECTIONAL BTC signal (executable on MT5).

Mechanism: funding = crowding gauge. Extreme positive funding => over-leveraged
longs => liquidation-cascade risk => bearish. Deeply negative => short capitulation
=> bounce. We can't COLLECT funding on MT5, but we can READ it (free, Binance)
and trade BTC direction.

Honesty rules: signal at day t uses only data through t; trades earn day t+1
onward. Pre-specified thresholds (z +/-2, hold 5d), tested ONCE. Also show the
conditional-returns table first — if forward returns aren't monotonic-ish across
funding buckets, there is no effect and no strategy tuning is allowed.
"""
import os, sys, time
import numpy as np
import pandas as pd
import ccxt

sys.path.insert(0, os.path.dirname(__file__))
from funding_carry_deep import fetch_funding          # Binance funding, paginated

START = "2021-01-01"
FEE_BPS = 10          # round-trip cost per position change (CFD spread-ish)
HOLD = 5              # days to hold after an extreme trigger (pre-specified)
Z_ENTRY = 2.0         # |z| threshold (pre-specified)


def fetch_btc_daily(start=START):
    ex = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
    since = ex.parse8601(start + "T00:00:00Z")
    rows, cursor = [], since
    while True:
        batch = ex.fetch_ohlcv("BTC/USDT", "1d", since=cursor, limit=1000)
        if not batch:
            break
        rows += batch
        cursor = batch[-1][0] + 86400_000
        if len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    return df.drop_duplicates("date").set_index("date")["close"]


def main():
    print("Fetching BTC daily closes + funding history (Binance, since 2021)...")
    close = fetch_btc_daily()
    fund8h = fetch_funding("BTC")
    fund_d = fund8h.resample("1D").sum()                     # daily funding (3 prints)
    df = pd.DataFrame({"close": close, "fund": fund_d}).dropna()
    print(f"{len(df)} days ({df.index[0].date()} -> {df.index[-1].date()})\n")

    # trailing 3d funding, z-scored vs trailing 90d (all backward-looking)
    f3 = df["fund"].rolling(3).mean()
    z = (f3 - f3.rolling(90).mean()) / f3.rolling(90).std()
    df["z"] = z

    # ---- 1) conditional forward returns by funding quintile ----
    ret = df["close"].pct_change()
    print("Forward BTC returns by trailing-funding quintile (Q1=most negative):")
    print(f"  {'quintile':<10}{'n':>6}{'fwd 1d':>9}{'fwd 3d':>9}{'fwd 7d':>9}")
    print("  " + "-" * 43)
    q = pd.qcut(df["z"], 5, labels=False)
    for k in range(5):
        m = q == k
        f1 = ret.shift(-1)[m].mean() * 100
        f3d = df["close"].pct_change(3).shift(-3)[m].mean() * 100
        f7d = df["close"].pct_change(7).shift(-7)[m].mean() * 100
        print(f"  Q{k+1:<9}{m.sum():>6}{f1:>8.2f}%{f3d:>8.2f}%{f7d:>8.2f}%")

    # ---- 2) pre-specified strategy: fade the extremes ----
    pos = pd.Series(0.0, index=df.index)
    state, days_left = 0.0, 0
    for i in range(len(df)):
        zi = df["z"].iloc[i]
        if days_left > 0:
            days_left -= 1
        else:
            state = 0.0
        if not np.isnan(zi):
            if zi > Z_ENTRY:
                state, days_left = -1.0, HOLD                # euphoric longs -> short
            elif zi < -Z_ENTRY:
                state, days_left = +1.0, HOLD                # capitulation -> long
        pos.iloc[i] = state

    gross = pos.shift(1) * ret
    costs = pos.diff().abs().fillna(0) * (FEE_BPS / 1e4)
    net = (gross - costs).dropna()
    intrade = net[pos.shift(1) != 0]

    def s(r):
        r = r.dropna()
        if len(r) == 0 or r.std() == 0:
            return "n/a"
        eq = (1 + r).cumprod()
        sh = r.mean() / r.std() * np.sqrt(365)
        dd = (eq / eq.cummax() - 1).min()
        return f"ret {eq.iloc[-1]-1:+7.1%}  sharpe {sh:5.2f}  maxDD {dd:6.1%}  hit {(r>0).mean()*100:3.0f}%"

    n_trades = int((pos.diff().abs() > 0).sum() / 2)
    print(f"\nStrategy (|z|>{Z_ENTRY} fade, hold {HOLD}d, {FEE_BPS}bps costs, "
          f"in market {(pos!=0).mean()*100:.0f}% of days, ~{n_trades} trades):")
    print(f"  strategy : {s(net)}")
    print(f"  in-trade days only: {s(intrade)}")
    print(f"  BTC B&H  : {s(ret.dropna())}")

    # long vs short legs separately (mechanism says both should work)
    print(f"  long leg (z<-2) : {s(net[pos.shift(1) > 0])}")
    print(f"  short leg (z>+2): {s(net[pos.shift(1) < 0])}")


if __name__ == "__main__":
    main()
