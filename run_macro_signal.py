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

    print(f"Macro signal — {dt.datetime.now(dt.timezone.utc).date().isoformat()} — {len(universe)} instruments "
          f"(model {ms.MODEL}, grounded by web search)\n")
    DELAY = 15   # seconds between instruments — stay under the web-search rate limit
    credits_dead = False
    tilts, done = [], {}
    for i, s in enumerate(universe):
        if i:
            time.sleep(DELAY)
        try:
            t = ms.get_tilt(s, refresh=refresh, verbose=True)
            tilts.append(t); done[s] = t
        except ms.CreditsExhausted:
            # every further call is a guaranteed failure — stop the run NOW
            credits_dead = True
            print(f"\n  !! OUT OF ANTHROPIC CREDITS at {s} — aborting the run "
                  f"({len(done)}/{len(universe)} done, all paid results are saved).")
            break
        except Exception as e:                           # noqa: BLE001
            print(f"  ! {s}: {e}")

    # one retry pass for anything that didn't ground (usually a transient rate-limit)
    ungrounded = [s for s, t in done.items() if not t.grounded]
    if ungrounded and not credits_dead:
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
            except ms.CreditsExhausted:
                credits_dead = True
                print("\n  !! OUT OF ANTHROPIC CREDITS during retries — stopping.")
                break
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

    # merge into any same-day book already on disk — a partial/aborted run must
    # only ever ADD to what earlier paid calls produced, never wipe it
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))
            if prev.get("as_of") == dt.datetime.now(dt.timezone.utc).date().isoformat():
                book = {**prev.get("targets", {}), **book}
        except Exception:                                # noqa: BLE001
            pass

    n_g = sum(t.grounded for t in tilts)
    json.dump({"as_of": dt.datetime.now(dt.timezone.utc).date().isoformat(), "model": ms.MODEL, "targets": book},
              open(OUT, "w"), indent=2)
    print(f"\n  {n_g}/{len(tilts)} grounded. wrote {OUT} (grounded tilt × confidence; executor sizes from this)")
    print(f"  appended rows to {ms.LOG}")

    # honest money accounting — what THIS run's fresh (non-cache) calls cost, est.
    live = [t for t in tilts if t.source == "claude+websearch"]
    n_srch = sum(t.searches for t in live)
    in_tok = sum(getattr(t, "in_tokens", 0) for t in live)
    out_tok = sum(getattr(t, "out_tokens", 0) for t in live)
    usd = n_srch * 0.01 + in_tok * 3 / 1e6 + out_tok * 15 / 1e6
    print(f"  cost: {len(live)} fresh calls, {n_srch} web searches, "
          f"{in_tok:,} in / {out_tok:,} out tokens ≈ ${usd:.2f} this run")

    if credits_dead:
        from notify import notify
        notify("Quant: OUT OF API CREDITS",
               f"Run stopped early — {n_g}/{len(universe)} markets done today. "
               "Add credits at console.anthropic.com; the next run resumes free from cache.")
        sys.exit(2)   # fail the CI step loudly; the workflow still commits saved work

    print("\n  Forward-test: this is unproven. Log it daily, and after a few weeks score")
    print("  hit-rate (did the tilt's sign match the next-N-day move?) before risking capital.")


if __name__ == "__main__":
    main()
