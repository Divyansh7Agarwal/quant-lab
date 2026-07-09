"""Score the forward-test log: did each macro tilt's direction match what the
market actually did over its horizon?

Reads macro_signal_log.jsonl, joins each tilt to the realized forward return of
its instrument (via the Yahoo proxy), and reports directional hit-rate against
the pre-committed 52% bar — plus a conviction-weighted return proxy.

This is the go/no-go instrument. A tilt only counts once `horizon_days` have
actually elapsed; everything newer is reported as 'pending'. Run it weekly:

    python score_macro_signal.py
"""
import os, sys, json
import datetime as dt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import data
import macro_signal as ms

LOG = ms.LOG


def load_log():
    if not os.path.exists(LOG):
        return []
    rows = [json.loads(l) for l in open(LOG) if l.strip()]
    # keep the latest row per (symbol, as_of) in case of --refresh re-runs
    latest = {}
    for r in rows:
        latest[(r["symbol"], r["as_of"])] = r
    return list(latest.values())


def realized_return(symbol, as_of, horizon):
    """Return over `horizon` trading days starting at the close on/after as_of,
    or None if not enough days have elapsed yet."""
    proxy = ms.INSTRUMENTS[symbol]["yf"]
    df = data.get(proxy, period="2y")
    idx = df.index
    d = pd.Timestamp(as_of)
    pos = idx.searchsorted(d)                       # first bar on/after the signal date
    if pos >= len(idx):
        return None
    end = pos + horizon
    if end >= len(idx):
        return None                                 # horizon not elapsed → pending
    c0 = float(df["close"].iloc[pos])
    c1 = float(df["close"].iloc[end])
    return c1 / c0 - 1


def main():
    rows = load_log()
    if not rows:
        print(f"No signals logged yet ({LOG} is empty).")
        print("Run `python run_macro_signal.py` daily first — the scorer needs")
        print("tilts whose horizon has elapsed before it can grade anything.")
        return

    ungrounded = [r for r in rows if not r.get("grounded", False)]
    rows = [r for r in rows if r.get("grounded", False)]
    if ungrounded:
        print(f"(excluding {len(ungrounded)} ungrounded tilts — no live search — from scoring)\n")

    scored, pending = [], 0
    for r in rows:
        rr = realized_return(r["symbol"], r["as_of"], int(r.get("horizon_days", 7)))
        if rr is None:
            pending += 1
            continue
        sign = np.sign(r["tilt"])
        if sign == 0:
            continue                                # flat call — nothing to grade
        scored.append({
            "symbol": r["symbol"], "as_of": r["as_of"], "tilt": r["tilt"],
            "conf": r.get("confidence", 0.0), "fwd_ret": rr,
            "hit": int(np.sign(rr) == sign),
            "pnl": sign * rr,                        # direction-only PnL proxy
            "wpnl": r["tilt"] * r.get("confidence", 1.0) * rr,  # conviction-weighted
        })

    if not scored:
        print(f"{len(rows)} tilts logged, {pending} still pending (horizon not elapsed).")
        print("Nothing gradable yet — check back once the earliest tilts mature.")
        return

    df = pd.DataFrame(scored)
    n = len(df)
    hit = df["hit"].mean()
    pnl = df["pnl"].values
    # t-stat on per-trade direction PnL → rough significance of the edge
    tstat = pnl.mean() / (pnl.std() / np.sqrt(n)) if pnl.std() > 0 else 0.0

    print(f"Macro-signal forward score — {dt.datetime.now(dt.timezone.utc).date().isoformat()}")
    print(f"{n} graded tilts, {pending} pending\n")

    print(f"  {'symbol':<8}{'n':>4}{'hit%':>7}{'avgFwd%':>9}")
    print("  " + "-" * 28)
    for sym, g in df.groupby("symbol"):
        print(f"  {sym:<8}{len(g):>4}{g['hit'].mean()*100:>6.0f}%{g['fwd_ret'].mean()*100:>8.1f}%")
    print("  " + "-" * 28)

    print(f"\n  Overall directional hit-rate : {hit*100:.1f}%   (bar to beat: 52%)")
    print(f"  Avg per-trade direction PnL  : {pnl.mean()*100:+.2f}%  (t-stat {tstat:.2f})")
    print(f"  Conviction-weighted avg ret  : {df['wpnl'].mean()*100:+.2f}%")

    verdict = ("PROMISING — keep running, widen the sample" if hit > 0.52 and tstat > 1
               else "NO EDGE YET — needs more data or the idea is dead" if n >= 40
               else "TOO EARLY — need ~40+ graded tilts to judge")
    print(f"\n  Verdict: {verdict}")
    print("  (Need a few hundred tilts and t-stat > ~2 before risking real capital.)")


if __name__ == "__main__":
    main()
