"""Paper-trading engine — a real practice portfolio, marked to market daily.

Every weekday (after the signal run) this:
  1. reads today's calls (target_tilts.json: weight = lean x sureness per market),
  2. sizes positions: dollars = weight x current equity, gross capped at 100%,
  3. "trades" the difference at today's close price, paying 4bps per trade,
  4. marks every open position to today's prices,
  5. appends one row to paper_equity.jsonl -> the equity curve the dashboard plots.

Starts with $1,000 of practice money. No real orders anywhere. This is the same
reconcile-to-target logic that will later drive the MT5 demo — so validating it
here validates the future execution path too.
"""
import os, sys, json, time
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
import macro_signal as ms

QUANT = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(QUANT, "paper_state.json")
CURVE = os.path.join(QUANT, "paper_equity.jsonl")
TILTS = os.path.join(QUANT, "target_tilts.json")

START_EQUITY = 1000.0
GROSS_CAP = 1.0          # at most 100% of equity at work (long + short combined)
COST_BPS = 4             # per trade, on traded dollars


def price_of(sym):
    proxy = ms.INSTRUMENTS[sym]["yf"]
    return float(data.get(proxy, period="1y")["close"].iloc[-1])


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"cash": START_EQUITY, "positions": {}, "started": dt.date.today().isoformat()}


def main():
    today = dt.date.today().isoformat()
    st = load_state()

    # skip if already run today (idempotent for re-runs)
    if os.path.exists(CURVE):
        rows = [json.loads(l) for l in open(CURVE) if l.strip()]
        if rows and rows[-1]["date"] == today:
            print(f"paper: already marked today (equity ${rows[-1]['equity']:.2f})")
            return

    # current prices for everything we hold or might trade
    tgt = {}
    if os.path.exists(TILTS):
        d = json.load(open(TILTS))
        tgt = d.get("targets", {})
    symbols = sorted(set(list(st["positions"].keys()) + list(tgt.keys())))
    prices = {}
    for s in symbols:
        try:
            prices[s] = price_of(s)
        except Exception as e:                    # noqa: BLE001
            print(f"paper: no price for {s} ({e}) — holding as-is")

    # mark to market
    pos_value = sum(p["units"] * prices.get(s, p["last_price"])
                    for s, p in st["positions"].items())
    equity = st["cash"] + pos_value

    # target dollars per market (gross-capped)
    gross = sum(abs(w) for w in tgt.values())
    scale = min(1.0, GROSS_CAP / gross) if gross > 0 else 0.0
    targets = {s: w * scale * equity for s, w in tgt.items()}

    # reconcile positions -> targets, pay costs on turnover
    costs = 0.0
    for s in symbols:
        if s not in prices:
            continue
        px = prices[s]
        cur_units = st["positions"].get(s, {}).get("units", 0.0)
        tgt_units = targets.get(s, 0.0) / px
        delta_units = tgt_units - cur_units
        trade_dollars = abs(delta_units) * px
        if trade_dollars < 1.0:                   # ignore dust
            if s in st["positions"]:
                st["positions"][s]["last_price"] = px
            continue
        costs += trade_dollars * COST_BPS / 1e4
        st["cash"] -= delta_units * px            # buy uses cash; sell/short adds
        if abs(tgt_units) * px < 1.0:
            st["positions"].pop(s, None)
        else:
            st["positions"][s] = {"units": tgt_units, "last_price": px}
    st["cash"] -= costs

    pos_value = sum(p["units"] * prices.get(s, p["last_price"])
                    for s, p in st["positions"].items())
    equity = st["cash"] + pos_value
    gross_now = sum(abs(p["units"]) * prices.get(s, p["last_price"])
                    for s, p in st["positions"].items())

    json.dump(st, open(STATE, "w"), indent=1)
    row = {"date": today, "equity": round(equity, 2), "cash": round(st["cash"], 2),
           "gross": round(gross_now, 2), "costs_today": round(costs, 2),
           "positions": {s: {"units": round(p["units"], 6),
                             "price": round(prices.get(s, p["last_price"]), 4),
                             "value": round(p["units"] * prices.get(s, p["last_price"]), 2)}
                         for s, p in st["positions"].items()},
           "ts": time.time()}
    with open(CURVE, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    pnl = equity - START_EQUITY
    print(f"paper: {today} equity ${equity:.2f} ({'+' if pnl>=0 else ''}{pnl:.2f} since start) "
          f"| ${gross_now:.0f} at work | costs today ${costs:.2f} | {len(st['positions'])} positions")


if __name__ == "__main__":
    main()
