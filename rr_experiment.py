"""The risk:reward experiment — does choosing a good R:R create edge?

Take real BTC daily data. Enter LONG on every single day (no signal, no skill).
Exit at a fixed stop or target. Try three risk:reward profiles:
    1:3  (risk 2%, target 6%)  — the classic "good R:R"
    1:1  (risk 3%, target 3%)
    3:1  (risk 6%, target 2%)  — "bad R:R", high win rate
Max hold 60 days, then exit at market. Costs 10bps round trip.

If R:R itself created edge, 1:3 should outperform. The math says it can't:
in a market without predictive structure, P(hit target before stop) ~ stop/(stop+target),
so the win rate mechanically falls exactly as much as the payoff rises. Exit rules
reshape the DISTRIBUTION of outcomes; they cannot move the MEAN. Only edge moves the mean.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import crypto_data as cd

COST = 0.001    # 10bps round trip
MAX_HOLD = 60


def run(close, stop, target):
    outcomes, holds = [], []
    n = len(close)
    for i in range(n - 1):
        e = close[i]
        out = None
        for j in range(i + 1, min(i + 1 + MAX_HOLD, n)):
            r = close[j] / e - 1
            if r <= -stop:
                out, h = -stop, j - i          # stopped (approx at stop level)
                break
            if r >= target:
                out, h = target, j - i          # target hit
                break
        if out is None:
            j = min(i + MAX_HOLD, n - 1)
            out, h = close[j] / e - 1, j - i    # time exit
        outcomes.append(out - COST)
        holds.append(h)
    o = np.array(outcomes)
    wins = o > 0
    return dict(n=len(o), win=wins.mean(),
                avg_win=o[wins].mean() if wins.any() else 0,
                avg_loss=o[~wins].mean() if (~wins).any() else 0,
                expectancy=o.mean(), hold=np.mean(holds))


def main():
    df = cd.get("BTC", days=1200)
    close = df["close"].values
    drift = (close[-1] / close[0]) ** (1 / len(close)) - 1    # daily drift of the asset
    print(f"BTC daily, {len(close)} days. Baseline daily drift {drift*100:+.3f}%/day "
          f"(the asset's own trend — 'beta').\n")
    print(f"  {'profile':<22}{'trades':>7}{'win%':>7}{'avgWin':>8}{'avgLoss':>9}"
          f"{'expect/trade':>13}{'per day held':>13}")
    print("  " + "-" * 79)
    for label, stop, target in [("1:3  (stop2% tgt6%)", 0.02, 0.06),
                                ("1:1  (stop3% tgt3%)", 0.03, 0.03),
                                ("3:1  (stop6% tgt2%)", 0.06, 0.02)]:
        s = run(close, stop, target)
        per_day = s["expectancy"] / s["hold"]
        print(f"  {label:<22}{s['n']:>7}{s['win']*100:>6.0f}%{s['avg_win']*100:>7.2f}%"
              f"{s['avg_loss']*100:>8.2f}%{s['expectancy']*100:>12.2f}%{per_day*100:>12.3f}%")
    print(f"\n  buy&hold same days: {drift*100:+.3f}%/day — the drift every profile inherits.")


if __name__ == "__main__":
    main()
