"""Daily macro-signal run: produce a Claude tilt for every instrument, print the
book, append to the JSONL log, and write target_tilts.json for the execution box.

Runs anywhere with internet + ANTHROPIC_API_KEY (this is the *thinking* half).
The MT5/Windows box reads target_tilts.json and trades it (the *acting* half).

  python run_macro_signal.py            # full book
  python run_macro_signal.py XAUUSD EURUSD   # subset
  python run_macro_signal.py --refresh  # ignore today's cache, re-query
"""
import os, sys, json, time
import datetime as dt

sys.path.insert(0, os.path.dirname(__file__))
import macro_signal as ms

OUT = os.path.join(os.path.dirname(__file__), "target_tilts.json")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refresh = "--refresh" in sys.argv
    universe = args or list(ms.INSTRUMENTS)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — cannot query Claude.\n")
        print("To run the live signal:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print(f"  python run_macro_signal.py\n")
        print("Dry preview — the price-action context each call would send to Claude:")
        for s in universe:
            try:
                print(f"  {s:<7} {ms.price_context(s)}")
            except Exception as e:                       # noqa: BLE001
                print(f"  {s:<7} (context failed: {e})")
        return

    print(f"Macro signal — {dt.date.today().isoformat()} — {len(universe)} instruments "
          f"(model {ms.MODEL}, grounded by web search)\n")
    DELAY = 25   # seconds between instruments — stay under the web-search rate limit
    tilts, done = [], {}
    for i, s in enumerate(universe):
        if i:
            time.sleep(DELAY)
        try:
            t = ms.get_tilt(s, refresh=refresh, verbose=True)
            tilts.append(t); done[s] = t
        except Exception as e:                           # noqa: BLE001
            print(f"  ! {s}: {e}")

    # one retry pass for anything that didn't ground (usually a transient rate-limit)
    ungrounded = [s for s, t in done.items() if not t.grounded]
    if ungrounded:
        print(f"\n  {len(ungrounded)} not grounded ({', '.join(ungrounded)}); "
              f"cooling down 60s then retrying once...")
        time.sleep(60)
        for i, s in enumerate(ungrounded):
            if i:
                time.sleep(DELAY)
            try:
                t = ms.get_tilt(s, refresh=True, verbose=True)
                if t.grounded:                           # replace the ungrounded row
                    tilts = [x for x in tilts if x.symbol != s] + [t]
            except Exception as e:                       # noqa: BLE001
                print(f"  ! retry {s}: {e}")

    print(f"\n  {'symbol':<8}{'stance':<7}{'tilt':>7}{'conf':>7}{'horizon':>8}{'grounded':>10}")
    print("  " + "-" * 48)
    book = {}
    for t in sorted(tilts, key=lambda x: -abs(x.tilt)):
        g = "yes" if t.grounded else "NO"
        print(f"  {t.symbol:<8}{t.stance:<7}{t.tilt:>+7.2f}{t.confidence:>7.2f}{t.horizon_days:>7}d{g:>10}")
        if t.grounded:                       # only trade signals that actually saw the news
            book[t.symbol] = round(t.tilt * t.confidence, 4)

    n_g = sum(t.grounded for t in tilts)
    json.dump({"as_of": dt.date.today().isoformat(), "model": ms.MODEL, "targets": book},
              open(OUT, "w"), indent=2)
    print(f"\n  {n_g}/{len(tilts)} grounded. wrote {OUT} (grounded tilt × confidence; executor sizes from this)")
    print(f"  appended rows to {ms.LOG}")
    print("\n  Forward-test: this is unproven. Log it daily, and after a few weeks score")
    print("  hit-rate (did the tilt's sign match the next-N-day move?) before risking capital.")


if __name__ == "__main__":
    main()
