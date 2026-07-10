"""Quant Lab dashboard — local backend.

Zero-dependency (stdlib http.server) API over the live logs in ~/quant:
  GET /              -> the UI (index.html)
  GET /api/overview  -> forward-test status, funding gauge, cost estimates
  GET /api/tilts     -> every logged tilt with full rationale
  GET /api/score     -> grades matured tilts vs realized market moves (slow first call)
  GET /api/funding/live -> recompute BTC funding z from Binance now (~15s)
  GET /api/graveyard -> every idea tested, verdict + lesson
  GET /api/ops       -> automation health, per-day cost log

Run:  python3 ~/quant/dashboard/server.py   ->  http://127.0.0.1:8765
Reads live files; no state of its own. Ctrl-C to stop.
"""
import os, sys, json, subprocess
import datetime as dt
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
QUANT = os.path.dirname(HERE)
sys.path.insert(0, QUANT)
import budget as _budget

MACRO_LOG = os.path.join(QUANT, "macro_signal_log.jsonl")
FUND_LOG = os.path.join(QUANT, "btc_funding_log.jsonl")
CRON_LOG = os.path.join(QUANT, "macro_cron.log")
PORT = 8765
BAR = 0.52
UNIVERSE_SIZE = 10


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def tilts():
    rows = _read_jsonl(MACRO_LOG)
    latest = {}
    for r in rows:                       # keep latest per (symbol, day) — refresh reruns
        latest[(r["symbol"], r["as_of"])] = r
    return sorted(latest.values(), key=lambda r: (r["as_of"], r["symbol"]))


def cost_estimate(day_rows):
    # real token+search accounting (budget.est_cost); rows logged before
    # 2026-07-10 have no token fields and fall back to a flat per-call estimate
    est = sum(_budget.est_cost(r) if r.get("in_tokens") else 0.45
              for r in day_rows if r.get("source") == "claude+websearch")
    return round(est, 2)


def overview():
    rows = tilts()
    days = sorted({r["as_of"] for r in rows})
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    todays = [r for r in rows if r["as_of"] == today] or ([r for r in rows if r["as_of"] == days[-1]] if days else [])
    grounded = [r for r in rows if r.get("grounded")]
    fund = _read_jsonl(FUND_LOG)
    day1 = "2026-07-03"
    trading_days = len([d for d in days if d >= day1])
    verdict_eta = "~Jul 24-31 (needs ~40 graded tilts)"
    return {
        "day1": day1, "days_logged": trading_days, "calendar_days": days,
        "tilts_total": len(rows), "tilts_grounded": len(grounded),
        "todays_book": todays, "todays_date": todays[0]["as_of"] if todays else None,
        "book_complete": len(todays) >= UNIVERSE_SIZE,
        "missing_today": UNIVERSE_SIZE - len(todays),
        "bar": BAR, "verdict_eta": verdict_eta,
        "funding": fund[-1] if fund else None,
        "cost_today": cost_estimate(todays),
        "model": "claude-sonnet-4-6", "search_cap": "~15 searches/call measured (~$0.47/market)",
    }


def score():
    import score_macro_signal as sc
    import numpy as np
    rows = [r for r in tilts() if r.get("grounded")]
    graded, pending = [], []
    for r in rows:
        try:
            rr = sc.realized_return(r["symbol"], r["as_of"], int(r.get("horizon_days", 7)))
        except Exception as e:                    # noqa: BLE001
            pending.append({**r, "note": f"data error: {e}"}); continue
        sign = 1 if r["tilt"] > 0 else (-1 if r["tilt"] < 0 else 0)
        if rr is None:
            due = (dt.date.fromisoformat(r["as_of"]) + dt.timedelta(days=int(r.get("horizon_days", 7)) * 1.45))
            pending.append({"symbol": r["symbol"], "as_of": r["as_of"], "stance": r["stance"],
                            "tilt": r["tilt"], "matures": due.isoformat()})
            continue
        if sign == 0:
            continue
        graded.append({"symbol": r["symbol"], "as_of": r["as_of"], "stance": r["stance"],
                       "tilt": r["tilt"], "fwd_ret": round(rr * 100, 2),
                       "hit": bool((rr > 0) == (sign > 0))})
    hits = [g["hit"] for g in graded]
    hit_rate = sum(hits) / len(hits) if hits else None
    # practice account: $1,000 start; each call puts $100 behind its direction,
    # P&L = market move x direction, minus ~$0.06 costs per trade
    equity = 1000.0
    for g in graded:
        d = 1 if g["tilt"] > 0 else -1
        g["pnl_usd"] = round(g["fwd_ret"] * d - 0.06, 2)
        equity += g["pnl_usd"]
    tstat = None
    if len(hits) >= 5:
        arr = np.array([1.0 if h else -1.0 for h in hits])
        tstat = round(float(arr.mean() / (arr.std() / max(len(arr), 1) ** 0.5)), 2) if arr.std() > 0 else None
    by_sym = {}
    for g in graded:
        s = by_sym.setdefault(g["symbol"], {"n": 0, "right": 0, "pnl": 0.0})
        s["n"] += 1; s["right"] += int(g["hit"]); s["pnl"] += g["pnl_usd"]
    per_symbol = [{"symbol": k, "n": v["n"], "hit_rate": round(v["right"] / v["n"], 3),
                   "pnl": round(v["pnl"], 2)}
                  for k, v in sorted(by_sym.items(), key=lambda kv: -kv[1]["n"])]
    return {"graded": graded, "pending": pending, "n_graded": len(graded),
            "hit_rate": hit_rate, "bar": BAR, "tstat": tstat,
            "practice_equity": round(equity, 2), "practice_start": 1000,
            "per_symbol": per_symbol,
            "flat_skipped": len([r for r in rows if r["tilt"] == 0])}


def funding_live():
    from funding_carry_deep import fetch_funding
    start = (dt.date.today() - dt.timedelta(days=300)).isoformat()
    f8h = fetch_funding("BTC", start=start)
    fd = f8h.resample("1D").sum()
    f3 = fd.rolling(3).mean()
    z = float(((f3 - f3.rolling(90).mean()) / f3.rolling(90).std()).iloc[-1])
    return {"date": dt.date.today().isoformat(), "z": round(z, 3),
            "trigger": "LONG" if z < -2 else ("SHORT_ZONE" if z > 2 else "none"),
            "fund_3d_pct_per_day": round(float(f3.iloc[-1]) * 100, 4)}


def paper():
    """The live practice portfolio: equity curve + current holdings."""
    curve_path = os.path.join(QUANT, "paper_equity.jsonl")
    if not os.path.exists(curve_path):
        return {"curve": [], "note": "practice account starts with the next daily run"}
    rows = [json.loads(l) for l in open(curve_path) if l.strip()]
    latest = rows[-1]
    holdings = [{"symbol": s, "direction": "UP" if v["units"] > 0 else "DOWN",
                 "dollars": abs(v["value"]), "price": v["price"]}
                for s, v in sorted(latest.get("positions", {}).items(),
                                   key=lambda kv: -abs(kv[1]["value"]))]
    # benchmark: $1,000 that just bought-and-held Bitcoin on the same start date
    bench = []
    try:
        import data as _data
        btc = _data.get("BTC-USD", period="1y")["close"]
        import pandas as pd
        px = {d.date().isoformat(): float(v) for d, v in btc.items()}
        dates = [r["date"] for r in rows]
        p0 = next((px[d] for d in dates if d in px), None)
        if p0:
            last = 1000.0
            for d in dates:
                if d in px:
                    last = round(1000.0 * px[d] / p0, 2)
                bench.append({"date": d, "equity": last})
    except Exception:                             # noqa: BLE001
        bench = []
    return {"curve": [{"date": r["date"], "equity": r["equity"]} for r in rows],
            "benchmark": bench,
            "start": 1000, "equity": latest["equity"],
            "pnl": round(latest["equity"] - 1000, 2),
            "at_work": latest.get("gross", 0), "cash": latest.get("cash", 0),
            "holdings": holdings, "as_of": latest["date"]}


def plan(capital=1000.0):
    """What a practice account would hold today, from the latest signal file."""
    path = os.path.join(QUANT, "target_tilts.json")
    if not os.path.exists(path):
        return {"positions": [], "note": "no signal file yet — runs after the daily 16:30 job"}
    d = json.load(open(path))
    import macro_signal as _ms
    pos = []
    for sym, w in sorted(d.get("targets", {}).items(), key=lambda kv: -abs(kv[1])):
        dollars = round(w * capital, 2)
        if abs(dollars) < 1:
            continue
        broker_sym = _ms.INSTRUMENTS.get(sym, {}).get("mt5", "not tradeable")
        pos.append({"symbol": sym, "broker_symbol": broker_sym,
                    "direction": "UP" if w > 0 else "DOWN",
                    "weight_pct": round(abs(w) * 100, 1), "dollars": abs(dollars)})
    total = round(sum(p["dollars"] for p in pos), 2)
    return {"as_of": d.get("as_of"), "capital": capital, "positions": pos,
            "total_at_work": total, "cash": round(capital - total, 2),
            "note": "Practice money. Position size = signal strength x confidence. Nothing is actually traded."}


GRAVEYARD = [
    dict(name="Kronos — AI that predicts price charts", verdict="DIDN'T WORK", cls="dead",
         metric="Right 49 times out of 100 — same as a coin flip",
         lesson="An AI trained on past prices couldn't predict future ones. Making the AI bigger made it worse."),
    dict(name="Trend following — ride what's going up", verdict="DIDN'T WORK", cls="dead",
         metric="Made less than simply buying and holding, with more stress",
         lesson="The classic 'follow the trend' recipe hasn't paid in today's markets. Tuning the settings didn't fix it."),
    dict(name="Buy the strongest coins, bet against the weakest", verdict="DIDN'T WORK", cls="dead",
         metric="Lost 56% over 3 years while plain Bitcoin holding tripled",
         lesson="Winners didn't keep winning. What looked like skill in crypto was just a rising market."),
    dict(name="Buy the dips (mean reversion)", verdict="DIDN'T WORK", cls="dead",
         metric="Lost money in every version we tested",
         lesson="Buying whatever just fell only works if someone is paying you to take that risk. Nobody was."),
    dict(name="Collect the 'impatience fee' from crypto gamblers", verdict="WORKS — CAN'T USE YET", cls="parked",
         metric="Would have made ~51%/yr with small dips, even through the 2022 crash",
         lesson="The one genuinely real money-maker found: get paid to hold, not to predict. Parked because your accounts can't reach it (exchange rules + Indian tax)."),
    dict(name="Bet against euphoric crowds", verdict="DIDN'T WORK", cls="dead",
         metric="Barely broke even after costs",
         lesson="Betting against an excited crowd means getting run over before you're proven right."),
    dict(name="Buy when the crowd panics", verdict="PROMISING — WATCHING", cls="live",
         metric="Good results, but it only happened ~15 times in 5 years",
         lesson="Too few examples to trust yet. We log it every day and buy the practice account in when it fires."),
    dict(name="Free-lunch trades at your broker", verdict="DIDN'T WORK", cls="dead",
         metric="The 4.7%/yr gold 'rent' is real — but the broker doesn't sell both halves of the trade",
         lesson="Every time we found real free income, the missing piece was the account needed to collect it."),
    dict(name="Sell insurance on the Indian market (options)", verdict="DIDN'T WORK", cls="dead",
         metric="Won 75% of months, still lost 6%/yr — the losses were huge",
         lesson="The insurance premium is real, but it goes to professional dealers, not to people holding to expiry. This is why 90% of Indian option traders lose."),
    dict(name="'Just use a good risk-reward ratio'", verdict="LESSON", cls="lesson",
         metric="The 'best' ratio earned exactly what Bitcoin itself earned — nothing extra",
         lesson="Stops and targets change how you win and lose, never how much on average. Only real predictions or real premiums do."),
    dict(name="Claude reads the news and calls direction", verdict="BEING TESTED NOW", cls="testing",
         metric="Live test started Jul 3. Needs to be right >52 times out of 100",
         lesson="The only idea that can't be tested on the past (Claude already knows the past) — so it's earning its grade live."),
]


def ops():
    cron_tail = []
    if os.path.exists(CRON_LOG):
        cron_tail = open(CRON_LOG, errors="ignore").read().splitlines()[-12:]
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=5).stdout
        job = next((l for l in out.splitlines() if "macrosignal" in l), None)
    except Exception:                             # noqa: BLE001
        job = None
    rows = tilts()
    by_day = {}
    for r in rows:
        by_day.setdefault(r["as_of"], []).append(r)
    costs = [{"date": d, "calls": len(v), "searches": sum(x.get("searches", 0) for x in v),
              "est_usd": cost_estimate(v)} for d, v in sorted(by_day.items())]
    latest_day = costs[-1]["date"] if costs else None
    utc_today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    return {"mode": "cloud", "launchd_loaded": bool(job), "launchd_line": job,
            "schedule": "GitHub Actions, weekdays — attempts 10:53/11:47/12:41 UTC + watchdog 15:19 UTC",
            "ran_today": latest_day == utc_today, "latest_run_day": latest_day,
            "budget": _budget.status(),
            "workdir": QUANT, "cron_tail": cron_tail, "costs_by_day": costs}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, *a):                    # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = self.path.split("?")[0]
        try:
            if route == "/api/overview":
                return self._json(overview())
            if route == "/api/tilts":
                return self._json({"tilts": tilts()})
            if route == "/api/score":
                return self._json(score())
            if route == "/api/funding/live":
                return self._json(funding_live())
            if route == "/api/graveyard":
                return self._json({"ideas": GRAVEYARD})
            if route == "/api/plan":
                return self._json(plan())
            if route == "/api/paper":
                return self._json(paper())
            if route == "/api/ops":
                return self._json(ops())
        except Exception as e:                    # noqa: BLE001
            return self._json({"error": str(e)}, 500)
        if route == "/":
            self.path = "/index.html"
        return super().do_GET()


if __name__ == "__main__":
    print(f"Quant Lab dashboard -> http://127.0.0.1:{PORT}   (reading {QUANT})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
