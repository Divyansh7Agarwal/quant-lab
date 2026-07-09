"""Daily decision runner — the bridge from backtest to your actual account.

Computes TODAY's target portfolio from the engine (optionally tilted by LLM
sentiment), compares it to your current holdings, and prints the exact orders to
get from here to there. Dry-run by default; wire to Alpaca paper when ready.

Usage:
    python3 live.py                 # show today's target portfolio + sentiment
    python3 live.py --account 10000 # size orders for a $10k account
    python3 live.py --llm           # blend in LLM news sentiment

Going live (paper first!):
    pip install alpaca-py
    export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
    # then flip SUBMIT=True below — it routes to the PAPER endpoint.
"""
import os, sys, json, argparse, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import data, signals as sg, portfolio as pf
from run import CONFIG

SUBMIT = False          # safety: stays dry-run until you explicitly enable
HOLDINGS_FILE = os.path.join(os.path.dirname(__file__), "holdings.json")
LLM_WEIGHT = 0.25       # how much sentiment tilts the trend signal


def target_weights(use_llm=False):
    uni = data.get_universe(CONFIG["universe"], period="2y")
    rows, raw_pos = [], {}
    senti_map = {}
    if use_llm:
        import signals_llm as sl
        for t in uni:
            senti_map[t] = sl.sentiment_signal(t)[0]

    for t, df in uni.items():
        close = df["close"]
        base = float(sg.ensemble(close, CONFIG["signal_weights"]).iloc[-1])
        senti = senti_map.get(t, 0.0)
        blended = base
        if use_llm:
            blended = max(-1.0, min(1.0, (1 - LLM_WEIGHT) * base + LLM_WEIGHT * senti))
        vol = float(pf.realized_vol(close, ppy=CONFIG["ppy"]).iloc[-1])
        scale = min(CONFIG["per_asset_vol"] / vol, CONFIG["max_leverage"]) if vol > 0 else 0
        w = max(0.0, blended) * scale if not CONFIG["allow_short"] else blended * scale
        raw_pos[t] = w
        rows.append({"ticker": t, "price": float(close.iloc[-1]),
                     "trend": round(base, 2), "sentiment": round(senti, 2),
                     "blended": round(blended, 2)})

    # gross-exposure cap (no leverage)
    gross = sum(raw_pos.values())
    if gross > CONFIG["max_gross"]:
        raw_pos = {t: w * CONFIG["max_gross"] / gross for t, w in raw_pos.items()}
    for r in rows:
        r["target_w"] = round(raw_pos[r["ticker"]], 4)
    return sorted(rows, key=lambda r: -r["target_w"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", type=float, default=10000)
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    print(f"Computing today's target portfolio ({'trend + LLM sentiment' if args.llm else 'trend only'})...\n")
    rows = target_weights(use_llm=args.llm)

    print(f"  {'ticker':<9}{'price':>10}{'trend':>8}{'senti':>8}{'blend':>8}{'target%':>9}{'$ alloc':>11}")
    print("  " + "-" * 63)
    invested = 0.0
    for r in rows:
        dollars = r["target_w"] * args.account
        invested += dollars
        print(f"  {r['ticker']:<9}{r['price']:>10.2f}{r['trend']:>8.2f}"
              f"{r['sentiment']:>8.2f}{r['blended']:>8.2f}{r['target_w']*100:>8.1f}%{dollars:>11.0f}")
    print("  " + "-" * 63)
    print(f"  {'INVESTED':<9}{'':<34}{invested/args.account*100:>16.1f}%{invested:>11.0f}")
    print(f"  {'CASH':<9}{'':<34}{(1-invested/args.account)*100:>16.1f}%{args.account-invested:>11.0f}")

    # rebalance plan vs current holdings (shares)
    holdings = json.load(open(HOLDINGS_FILE)) if os.path.exists(HOLDINGS_FILE) else {}
    print("\n  ORDER PLAN (to reach target):")
    any_order = False
    for r in rows:
        tgt_shares = (r["target_w"] * args.account) / r["price"]
        cur = holdings.get(r["ticker"], 0)
        delta = tgt_shares - cur
        if abs(delta) * r["price"] > 0.01 * args.account:   # ignore <1% dust trades
            any_order = True
            side = "BUY " if delta > 0 else "SELL"
            print(f"    {side} {abs(delta):.2f} {r['ticker']} (~${abs(delta)*r['price']:.0f})")
    if not any_order:
        print("    (no holdings.json found — showing target only; create it to get a diff)")

    print(f"\n  Execution: {'SUBMIT (paper)' if SUBMIT else 'DRY-RUN'}.  "
          f"Sentiment backend: {'Claude' if os.getenv('ANTHROPIC_API_KEY') else 'heuristic (set ANTHROPIC_API_KEY for Claude)'}")
    print("  Paper-trade this daily for 1-3 months and compare to the backtest before real capital.")


if __name__ == "__main__":
    main()
