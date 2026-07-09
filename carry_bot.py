"""Paper funding-carry bot — Bybit TESTNET — BTC-only, perp-against-held-spot.

You already hold BTC (the long-spot leg). This bot manages only the PERP leg:
short a small BTC-perp notional when funding is positive (collect it), cover when
funding turns negative. Long spot + short perp = delta-neutral basis trade.

Decision uses REAL mainnet funding (testnet funding is synthetic); orders execute
on TESTNET. Dry-run by default; --submit places orders and still refuses unless
BYBIT_TESTNET=1. Small fixed size.
"""
import os, sys, json, time
import datetime as dt
import ccxt

COINS = ["BTC"]          # start with one coin
PERP_USDT = 500          # perp short notional (the carry leg) — small test size
TRAIL_PERIODS = 21       # ~7d of 8h funding for the on/off decision
LOG = os.path.join(os.path.dirname(__file__), "carry_log.jsonl")


def connect():
    """`ex` = authed TESTNET (positions/orders); `pub` = public MAINNET (REAL funding)."""
    k, s = os.environ.get("BYBIT_API_KEY"), os.environ.get("BYBIT_API_SECRET")
    if not (k and s):
        raise SystemExit("BYBIT_API_KEY / BYBIT_API_SECRET not set (check ~/.zshenv)")
    ex = ccxt.bybit({"apiKey": k, "secret": s, "enableRateLimit": True, "timeout": 15000,
                     "options": {"defaultType": "unified"}})
    if os.environ.get("BYBIT_TESTNET") == "1":
        ex.set_sandbox_mode(True)
    ex.load_markets()
    pub = ccxt.bybit({"enableRateLimit": True, "timeout": 15000, "options": {"defaultType": "swap"}})
    return ex, pub


def funding_state(pub, coin):
    """(current_rate, trailing_mean, want_on) from REAL mainnet funding."""
    hist = pub.fetch_funding_rate_history(f"{coin}/USDT:USDT", limit=TRAIL_PERIODS)
    rates = [float(h["fundingRate"]) for h in hist] or [0.0]
    trail = sum(rates) / len(rates)
    return rates[-1], trail, (trail > 0)


def book(ex, coin):
    """(spot_usdt_held, perp_usdt_signed, price) — perp negative = short."""
    price = ex.fetch_ticker(f"{coin}/USDT")["last"]
    spot_qty = float(ex.fetch_balance().get(coin, {}).get("total") or 0)
    perp_usdt = 0.0
    try:
        for p in ex.fetch_positions([f"{coin}/USDT:USDT"]):
            contracts = float(p.get("contracts") or 0)
            notional = float(p.get("notional") or contracts * price)
            perp_usdt = -notional if p.get("side") == "short" else notional
    except Exception as e:                       # noqa: BLE001
        print(f"    (positions read: {e})")
    return spot_qty * price, perp_usdt, price


def plan_for(pub, ex, coin):
    cur, trail, want_on = funding_state(pub, coin)
    spot_usdt, perp_usdt, price = book(ex, coin)
    tgt_perp = -PERP_USDT if want_on else 0.0
    return dict(coin=coin, price=price, funding_now=cur, funding_trail=trail, want_on=want_on,
                spot_now=spot_usdt, perp_now=perp_usdt, perp_target=tgt_perp,
                perp_delta=tgt_perp - perp_usdt,
                est_funding_per_period=abs(tgt_perp) * cur)


def execute(ex, p, submit):
    if not submit or abs(p["perp_delta"]) < 10:
        return
    perp_sym = f"{p['coin']}/USDT:USDT"
    side = "sell" if p["perp_delta"] < 0 else "buy"     # sell = open/extend short; buy = cover
    qty = round(abs(p["perp_delta"]) / p["price"], 6)
    try:
        ex.set_leverage(2, perp_sym)
    except Exception:                                   # noqa: BLE001
        pass
    r = ex.create_order(perp_sym, "market", side, qty,
                        params={"reduceOnly": p["perp_delta"] > 0})
    print(f"    PERP {side} {qty} {p['coin']} -> order {r.get('id')}")


def main():
    submit = "--submit" in sys.argv
    if submit and os.environ.get("BYBIT_TESTNET") != "1":
        raise SystemExit("REFUSING --submit: BYBIT_TESTNET != 1 (won't touch a real account)")

    ex, pub = connect()
    mode = "LIVE-SUBMIT (testnet)" if submit else "DRY-RUN"
    print(f"Carry bot — {dt.date.today().isoformat()} — {mode} — perp-against-held-BTC\n")

    rows = []
    for c in COINS:
        p = plan_for(pub, ex, c)
        tgt = f"SHORT ${PERP_USDT:.0f} perp" if p["want_on"] else "FLAT (funding<0, cover)"
        print(f"  {c}: real funding now {p['funding_now']*100:+.4f}%/8h, "
              f"trail {p['funding_trail']*100:+.4f}% -> target {tgt}")
        print(f"     spot held: ${p['spot_now']:.0f} (your long leg) | "
              f"perp now: ${p['perp_now']:.0f} -> Δ${p['perp_delta']:+.0f}")
        if p["want_on"]:
            print(f"     est carry on ${PERP_USDT}: ${p['est_funding_per_period']:+.4f}/8h "
                  f"(~${p['est_funding_per_period']*3*365:+.2f}/yr)")
        execute(ex, p, submit)
        rows.append(p)

    with open(LOG, "a") as fh:
        fh.write(json.dumps({"ts": time.time(), "date": dt.date.today().isoformat(),
                             "mode": mode, "plan": rows}) + "\n")
    print(f"\n  logged to {LOG}")
    if not submit:
        print("  DRY-RUN — nothing traded. Re-run with --submit (testnet) to place the short.")


if __name__ == "__main__":
    main()
