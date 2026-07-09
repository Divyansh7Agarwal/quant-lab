"""Daily decision + execution runner for MetaTrader 5.  RUN ON WINDOWS/VPS.

Pulls D1 data from MT5 for your tradeable symbols, computes today's target book
from the engine (trend ensemble, vol-targeted, gross-capped, long/short), converts
to lots, and reconciles your live positions to it.

SAFETY:
  * DRY_RUN = True by default — prints the order plan, places nothing.
  * Even with DRY_RUN off, it REFUSES to trade a non-demo account unless you pass
    --allow-live explicitly. Forward-test on DEMO for weeks before that flag.

Usage on the VPS (MT5 terminal running + logged into your DEMO account):
    python live_mt5.py                 # dry-run plan
    python live_mt5.py --submit        # execute on DEMO
    # schedule once/day after the daily close via Windows Task Scheduler.
"""
import os, sys, argparse, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import signals as sg
import portfolio as pf
import mt5_adapter as broker

DRY_RUN = True

# Syntex's ACTUAL symbols (verified in the terminal 2026-07-04). Metals/indices/oil are
# DATED futures CFDs that expire (Q26=Aug, U26=Sep) — rollover must be handled at expiry.
# Canonical signal->broker mapping lives in macro_signal.INSTRUMENTS[sym]["mt5"].
# NOTE: account is NETTING mode (one net position per symbol) — reconcile logic fits.
UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "BTCUSD",
            "GC-Q26", "SI-U26", "CL-Q26", "SP-U26", "NASDAQ-U26"]

CFG = {
    "per_asset_vol": 0.10,    # target ~10% annualized vol per instrument at full conviction
    "max_leverage": 0.5,      # cap any single instrument's weight
    "max_gross": 1.5,         # total gross notional cap (1.5x equity across the book)
    "allow_short": True,      # MT5 shorts freely — trend works both directions
    "ppy": 252,
    # slow-CTA: long lookbacks, no fast mean-reversion -> ~16x/yr turnover (robust to costs)
    "signal_weights": {"tsmom": 0.75, "xover": 0.25, "meanrev": 0.0},
    "mom_lookbacks": (60, 120, 250), "xover": (50, 200),
}


def target_book(universe_data, equity):
    raw = {}
    rows = []
    for sym, df in universe_data.items():
        close = df["close"]
        sig = float(sg.ensemble(close, CFG["signal_weights"],
                                mom_lookbacks=CFG["mom_lookbacks"],
                                xover=CFG["xover"]).iloc[-1])
        vol = float(pf.realized_vol(close, ppy=CFG["ppy"]).iloc[-1])
        scale = min(CFG["per_asset_vol"] / vol, CFG["max_leverage"]) if vol > 0 else 0.0
        w = sig * scale
        if not CFG["allow_short"]:
            w = max(0.0, w)
        w = max(-CFG["max_leverage"], min(CFG["max_leverage"], w))
        raw[sym] = w
        rows.append({"sym": sym, "signal": round(sig, 2), "price": float(close.iloc[-1])})

    gross = sum(abs(v) for v in raw.values())
    if gross > CFG["max_gross"]:
        raw = {s: v * CFG["max_gross"] / gross for s, v in raw.items()}
    for r in rows:
        r["weight"] = round(raw[r["sym"]], 4)
        r["lots"] = broker.lots_for_weight(r["sym"], raw[r["sym"]], equity)
    return sorted(rows, key=lambda r: -abs(r["weight"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="actually place orders")
    ap.add_argument("--allow-live", action="store_true", help="permit trading a NON-demo account")
    args = ap.parse_args()
    do_submit = args.submit and not DRY_RUN     # module DRY_RUN is the master lock

    acct = broker.connect()
    eq = acct["equity"]
    print(f"Connected: account {acct['login']} | {acct['currency']} {eq:,.2f} equity | "
          f"{'DEMO' if acct['is_demo'] else 'LIVE'} | leverage 1:{acct['leverage']}")

    data = broker.get_universe(UNIVERSE, timeframe="D1", n=800)
    print(f"Loaded {len(data)}/{len(UNIVERSE)} symbols\n")

    book = target_book(data, eq)
    cur = broker.net_positions()

    print(f"  {'symbol':<9}{'signal':>8}{'price':>11}{'weight':>9}{'tgt lots':>10}{'cur lots':>10}{'action':>9}")
    print("  " + "-" * 66)
    plan = []
    for r in book:
        c = cur.get(r["sym"], 0.0)
        delta = r["lots"] - c
        act = "flat" if abs(delta) < 0.01 else ("BUY" if delta > 0 else "SELL")
        if act != "flat":
            plan.append((r["sym"], r["lots"]))
        print(f"  {r['sym']:<9}{r['signal']:>8.2f}{r['price']:>11.4f}{r['weight']*100:>8.1f}%"
              f"{r['lots']:>10.2f}{c:>10.2f}{act:>9}")
    print("  " + "-" * 66)
    print(f"  gross target: {sum(abs(r['weight']) for r in book)*100:.0f}% of equity | {len(plan)} orders\n")

    if do_submit:
        if not acct["is_demo"] and not args.allow_live:
            print("REFUSING to trade a LIVE account without --allow-live. Forward-test on DEMO first.")
            broker.shutdown(); return
        for sym, lots in plan:
            res = broker.rebalance_symbol(sym, lots)
            print(f"  {sym}: {'OK' if res and res.retcode == 10009 else res}")
        print("\nSubmitted.")
    else:
        why = "DRY_RUN lock is on (edit DRY_RUN=False)" if (args.submit and DRY_RUN) \
              else "pass --submit (on DEMO) to execute"
        print(f"DRY-RUN — no orders sent. {why}.")
    broker.shutdown()


if __name__ == "__main__":
    main()
