"""Credit budget tracker — the single source of truth for what Claude calls cost.

budget.json records how many dollars of Anthropic credits were added and since
when. Spend is estimated from macro_signal_log.jsonl, where every live call logs
its billed searches and tokens. From those two, everything else derives:

  python budget.py status     # spent / remaining / runway in days
  python budget.py add 20     # user just topped up $20

The estimate can drift a few % from Anthropic's real meter (their rounding,
non-logged failures) — treat it as a fuel gauge, not an invoice.
"""
import os, sys, json
import datetime as dt

QUANT = os.path.dirname(os.path.abspath(__file__))
BUDGET = os.path.join(QUANT, "budget.json")
LOG = os.path.join(QUANT, "macro_signal_log.jsonl")

# claude-sonnet-4-6 list prices (cache writes +25%, cache reads -90%)
IN_PER_MTOK, OUT_PER_MTOK, PER_SEARCH = 3.0, 15.0, 0.01
CACHE_W_PER_MTOK, CACHE_R_PER_MTOK = 3.75, 0.30

# thresholds shared by runner + watchdog
WARN_REMAINING_USD = 10.0     # ~2 run-days: warn the user to top up
RUN_CAP_USD = 8.0             # no single run may spend more than this


def est_cost(row):
    """Estimated $ for one logged call (dict with searches/token fields)."""
    return (row.get("searches", 0) * PER_SEARCH
            + row.get("in_tokens", 0) * IN_PER_MTOK / 1e6
            + row.get("out_tokens", 0) * OUT_PER_MTOK / 1e6
            + row.get("cache_w_tokens", 0) * CACHE_W_PER_MTOK / 1e6
            + row.get("cache_r_tokens", 0) * CACHE_R_PER_MTOK / 1e6)


def _load():
    if os.path.exists(BUDGET):
        return json.load(open(BUDGET))
    return {"budget_usd": 0.0, "since": None, "history": []}


def status():
    b = _load()
    if not b.get("since"):
        return {"configured": False}
    since_ts = dt.datetime.fromisoformat(b["since"]).timestamp()
    rows = []
    if os.path.exists(LOG):
        for line in open(LOG):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("ts", 0) >= since_ts and r.get("source") == "claude+websearch":
                rows.append(r)
    spent = sum(est_cost(r) for r in rows)
    remaining = b["budget_usd"] - spent
    by_day = {}
    for r in rows:
        by_day.setdefault(r.get("as_of", "?"), []).append(r)
    day_costs = [sum(est_cost(r) for r in v) for _, v in sorted(by_day.items())]
    avg_day = (sum(day_costs[-3:]) / len(day_costs[-3:])) if day_costs else 0.0
    return {
        "configured": True, "budget_usd": round(b["budget_usd"], 2),
        "since": b["since"], "spent_usd": round(spent, 2),
        "remaining_usd": round(remaining, 2),
        "avg_day_usd": round(avg_day, 2),
        "runway_days": round(remaining / avg_day, 1) if avg_day > 0 else None,
        "low": remaining < WARN_REMAINING_USD,
    }


def add(usd):
    b = _load()
    now = dt.datetime.now(dt.timezone.utc)
    if not b.get("since"):
        b["since"] = now.isoformat(timespec="seconds")
    b["budget_usd"] = round(b.get("budget_usd", 0.0) + usd, 2)
    b.setdefault("history", []).append(
        {"date": now.date().isoformat(), "usd": usd})
    json.dump(b, open(BUDGET, "w"), indent=2)
    return b


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "add":
        b = add(float(sys.argv[2]))
        print(f"recorded top-up; total budget since {b['since']}: ${b['budget_usd']:.2f}")
    s = status()
    if not s.get("configured"):
        print("budget.json not initialised — run: python budget.py add <usd>")
    else:
        runway = f"{s['runway_days']} run-days" if s["runway_days"] is not None else "n/a (no runs yet)"
        print(f"credits: ${s['spent_usd']:.2f} spent of ${s['budget_usd']:.2f} since {s['since'][:10]} "
              f"→ ${s['remaining_usd']:.2f} left ≈ {runway}"
              + ("  ⚠ LOW — top up soon" if s["low"] else ""))
