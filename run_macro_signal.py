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
import budget

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
    DELAY = 15         # between serial retry calls only
    PARALLEL = 3       # markets bought simultaneously — a slow call can't block the rest
    credits_dead = cap_hit = False

    def run_cost(ts):
        return sum(budget.est_cost(t.__dict__) for t in ts if t.source == "claude+websearch")

    # prefetch price data serially first (yfinance + csv cache aren't thread-friendly;
    # the Claude calls afterwards are). Failures here resurface inside get_tilt.
    for s in universe:
        try:
            ms.price_context(s)
        except Exception:                                # noqa: BLE001
            pass

    from concurrent.futures import ThreadPoolExecutor, as_completed
    tilts, done = [], {}
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = {pool.submit(ms.get_tilt, s, refresh, True): s for s in universe}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                t = fut.result()
                tilts.append(t); done[s] = t
                if run_cost(tilts) > budget.RUN_CAP_USD and not cap_hit:
                    cap_hit = True
                    print(f"\n  !! COST CAP: this run is ≈${run_cost(tilts):.2f}, over the "
                          f"${budget.RUN_CAP_USD:.0f} safety cap — stopping. Billing is behaving "
                          "abnormally; investigate before the next run. All results are saved.")
                    pool.shutdown(cancel_futures=True)
                    break
            except ms.CreditsExhausted:
                # every further call is a guaranteed failure — stop the run NOW
                credits_dead = True
                print(f"\n  !! OUT OF ANTHROPIC CREDITS at {s} — aborting the run "
                      f"({len(done)}/{len(universe)} done, all paid results are saved).")
                pool.shutdown(cancel_futures=True)
                break
            except Exception as e:                       # noqa: BLE001
                print(f"  ! {s}: {e}")

    # one retry pass for anything that didn't ground (usually a transient rate-limit)
    ungrounded = [s for s, t in done.items() if not t.grounded]
    if ungrounded and not credits_dead and not cap_hit:
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
                if run_cost(tilts) > budget.RUN_CAP_USD:
                    cap_hit = True
                    print("\n  !! COST CAP exceeded during retries — stopping.")
                    break
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
    from notify import notify
    live = [t for t in tilts if t.source == "claude+websearch"]
    n_srch = sum(t.searches for t in live)
    in_tok = sum(getattr(t, "in_tokens", 0) for t in live)
    out_tok = sum(getattr(t, "out_tokens", 0) for t in live)
    usd = run_cost(tilts)
    print(f"  cost: {len(live)} fresh calls, {n_srch} web searches, "
          f"{in_tok:,} in / {out_tok:,} out tokens ≈ ${usd:.2f} this run")
    bs = budget.status()
    if bs.get("configured"):
        runway = (f"≈{bs['runway_days']} run-days" if bs["runway_days"] is not None else "n/a")
        print(f"  credits: ${bs['spent_usd']:.2f} used of ${bs['budget_usd']:.2f} "
              f"→ ${bs['remaining_usd']:.2f} left ({runway})")

    if credits_dead:
        notify("Quant: OUT OF API CREDITS",
               f"Run stopped early — {n_g}/{len(universe)} markets done today. "
               "Add credits at console.anthropic.com; the next run resumes free from cache.")
        sys.exit(2)   # fail the CI step loudly; the workflow still commits saved work
    if cap_hit:
        notify("Quant: run stopped by cost cap",
               f"This run hit the ${budget.RUN_CAP_USD:.0f} safety cap (≈${usd:.2f}) — "
               "billing is abnormal. All paid results saved; investigate before next run.")
        sys.exit(3)
    if bs.get("configured") and bs["low"]:
        notify("Quant: credits running low",
               f"≈${bs['remaining_usd']:.2f} left ({runway}). "
               "Top up at console.anthropic.com, then run: python budget.py add <usd>")
    if live:   # fresh work done today — positive confirmation (silent on cached re-runs)
        notify("Quant: today's book is in",
               f"{n_g}/{len(universe)} markets news-checked, ≈${usd:.2f} this run"
               + (f", ${bs['remaining_usd']:.0f} credits left" if bs.get("configured") else ""))

    print("\n  Forward-test: this is unproven. Log it daily, and after a few weeks score")
    print("  hit-rate (did the tilt's sign match the next-N-day move?) before risking capital.")


if __name__ == "__main__":
    main()
