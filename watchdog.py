"""End-of-day watchdog — answers one question: did today actually happen?

Runs on its own schedule AFTER all three pipeline attempts. Reads only committed
state (no API calls, no cost) and fails RED + notifies if anything a trading day
needs is missing. Green days are silent — this is an alarm, not a newsletter.

Checks, in plain language:
  1. Was today's book produced, and is it reasonably complete?
  2. Did the practice account mark today?
  3. Are credits about to run out?

Exit 0 = all good. Exit 1 = something needs a human. Weekends: no-op.
"""
import os, sys, json
import datetime as dt

QUANT = os.path.dirname(os.path.abspath(__file__))
TILTS = os.path.join(QUANT, "target_tilts.json")
CURVE = os.path.join(QUANT, "paper_equity.jsonl")

sys.path.insert(0, QUANT)
import budget
from notify import notify

UNIVERSE_SIZE = 10
MIN_BOOK = 8          # fewer news-checked markets than this = something's wrong


def main():
    now = dt.datetime.now(dt.timezone.utc)
    today = now.date().isoformat()
    if now.weekday() >= 5:
        print(f"watchdog: {today} is a weekend — nothing to check")
        return 0

    problems, notes = [], []

    # 1. today's book
    if not os.path.exists(TILTS):
        problems.append("no book file at all (target_tilts.json missing)")
    else:
        d = json.load(open(TILTS))
        n = len(d.get("targets", {}))
        if d.get("as_of") != today:
            problems.append(f"no book for today — newest is from {d.get('as_of')} "
                            "(all pipeline attempts were skipped or failed)")
        elif n < MIN_BOOK:
            problems.append(f"today's book has only {n}/{UNIVERSE_SIZE} markets "
                            "news-checked — most calls failed")
        elif n < UNIVERSE_SIZE:
            notes.append(f"book is {n}/{UNIVERSE_SIZE} (minor gaps)")
        else:
            notes.append(f"book complete {n}/{UNIVERSE_SIZE}")

    # 2. today's practice-account mark
    last = None
    if os.path.exists(CURVE):
        for line in open(CURVE):
            if line.strip():
                last = json.loads(line)
    if not last or last.get("date") != today:
        problems.append("practice account did not mark today "
                        f"(last mark: {last.get('date') if last else 'never'})")
    else:
        notes.append(f"practice equity ${last['equity']:.2f}")

    # 3. credit runway
    bs = budget.status()
    if bs.get("configured"):
        if bs["remaining_usd"] <= 0:
            problems.append(f"credits are GONE (≈${bs['remaining_usd']:.2f}) — "
                            "the test is stalled until you top up")
        elif bs["low"]:
            problems.append(f"credits low: ≈${bs['remaining_usd']:.2f} left "
                            f"(~{bs['runway_days']} run-days)")
        else:
            notes.append(f"credits ok (${bs['remaining_usd']:.2f} left)")

    if problems:
        msg = " · ".join(problems)
        print(f"watchdog: PROBLEMS — {msg}")
        notify("Quant watchdog: needs attention", msg)
        return 1
    print(f"watchdog: all good — {' · '.join(notes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
