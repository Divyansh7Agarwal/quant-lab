"""Macro-narrative signal — the candidate alpha layer for the MT5 book.

For each instrument, Claude reads the CURRENT market-moving context via live web
search (central-bank communications, economic-data surprises, risk sentiment,
geopolitics) plus a compact price-action summary, and returns a structured
directional tilt in [-1, 1] with its reasoning.

Why web search: it grounds the tilt in *today's* developments instead of the
model's training memory. That also means this signal CANNOT be honestly
backtested (the model knows history; search returns current info) — its only
valid validation path is FORWARD paper-trading. Tilts are cached per (symbol,
day) so re-runs don't re-pay, and every call is appended to a JSONL log so you
can score hit-rate going forward.

Needs ANTHROPIC_API_KEY in the environment. Runs anywhere with internet — only
the EXECUTION half (live_mt5.py) needs the Windows/MT5 box.
"""
import os, re, json, time, threading
import datetime as dt
from dataclasses import dataclass, asdict

import data  # reuse the yfinance price layer

# Sonnet 4.6 while the signal is UNPROVEN — ~2-5x cheaper than opus for a daily
# loop; promote back to claude-opus-4-8 only if it clears the 52% bar. web_search
# _20260209 (dynamic filtering) is supported on Sonnet 4.6.
MODEL = "claude-sonnet-4-6"
# max_uses resets on each pause_turn continuation, so real cap = max_uses x loop rounds.
# MEASURED cost per call (dynamic filtering bills many queries per tool-use, so this
# knob is a loose lid, not a budget — trust the searches/in_tokens logged per Tilt):
#   max_uses=3 → 16 searches            ≈ $0.50+
#   max_uses=2 → 15 searches,  93k in   ≈ $0.47   ← cheapest that grounds; KEEP
#   max_uses=1 → 26 searches, 205k in   ≈ $0.97   (starving rounds backfires — 2x cost)
WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search", "max_uses": 2}

CACHE = os.path.join(os.path.dirname(__file__), "cache", "macro")
LOG = os.path.join(os.path.dirname(__file__), "macro_signal_log.jsonl")
_log_lock = threading.Lock()   # get_tilt may run from parallel workers
os.makedirs(CACHE, exist_ok=True)

# Signal universe = ONLY what the broker (Syntex) can actually trade (user decision
# 2026-07-04). `mt5` = the real broker symbol; dated futures (…-Q26/-U26) expire and
# roll — the executor handles that, the signal always uses continuous Yahoo proxies.
INSTRUMENTS = {
    "XAUUSD": dict(yf="GC=F",    mt5="GC-Q26",     name="Gold (XAU/USD)",
                   drivers="real US yields, the USD, safe-haven demand, central-bank buying"),
    "XAGUSD": dict(yf="SI=F",    mt5="SI-U26",     name="Silver (XAG/USD)",
                   drivers="industrial demand (solar, EVs), gold correlation, real yields, the USD"),
    "EURUSD": dict(yf="EURUSD=X", mt5="EURUSD",    name="Euro vs US Dollar",
                   drivers="ECB-vs-Fed policy divergence, EZ-vs-US growth, rate differentials"),
    "GBPUSD": dict(yf="GBPUSD=X", mt5="GBPUSD",    name="British Pound vs US Dollar",
                   drivers="BoE policy, UK inflation/growth, risk sentiment"),
    "USDJPY": dict(yf="USDJPY=X", mt5="USDJPY",    name="US Dollar vs Japanese Yen",
                   drivers="US-Japan rate differentials, BoJ policy, risk-on/off flows"),
    "AUDUSD": dict(yf="AUDUSD=X", mt5="AUDUSD",    name="Australian Dollar vs US Dollar",
                   drivers="RBA-vs-Fed policy, China demand and stimulus, commodity prices, risk sentiment"),
    "US500":  dict(yf="ES=F",    mt5="SP-U26",     name="S&P 500 index",
                   drivers="US growth, Fed path, earnings, risk appetite"),
    "USTEC":  dict(yf="NQ=F",    mt5="NASDAQ-U26", name="Nasdaq-100 index",
                   drivers="tech earnings, real yields, AI/semis sentiment, Fed path"),
    "USOIL":  dict(yf="CL=F",    mt5="CL-Q26",     name="WTI Crude Oil",
                   drivers="OPEC+ supply, global demand, geopolitics, USD"),
    "BTCUSD": dict(yf="BTC-USD", mt5="BTCUSD",     name="Bitcoin vs US Dollar",
                   drivers="risk appetite, ETF/flows, liquidity, regulation, crypto-native news"),
}


@dataclass
class Tilt:
    symbol: str
    stance: str            # "long" | "short" | "flat"
    tilt: float            # -1.0 (max short) .. +1.0 (max long)
    confidence: float      # 0.0 .. 1.0
    horizon_days: int
    rationale: str
    key_drivers: list
    as_of: str
    model: str
    source: str            # "claude+websearch" | "cache"
    grounded: bool = False # did live search actually return results this call?
    searches: int = 0      # number of web searches Claude issued
    in_tokens: int = 0     # billed input tokens (cost audit trail)
    out_tokens: int = 0    # billed output tokens
    cache_w_tokens: int = 0  # prompt-cache writes (billed at 1.25x input)
    cache_r_tokens: int = 0  # prompt-cache reads  (billed at 0.10x input)


class CreditsExhausted(RuntimeError):
    """API says the account is out of credits — abort the whole run, don't
    march through the remaining symbols burning time on guaranteed failures."""


def _today():
    # UTC, everywhere. The Mac (IST) and the GitHub runner (UTC) must agree on
    # "today" or a market bought on one gets re-bought on the other near midnight.
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


# ----------------------- price-action context -----------------------
def price_context(symbol):
    """Compact, factual price summary so the model anchors on real levels."""
    meta = INSTRUMENTS[symbol]
    data.last_close(meta["yf"])   # raises SanityError on stale/absurd feed —
    df = data.get(meta["yf"], period="2y")   # better no call than a paid call
    c = df["close"]                          # anchored on garbage prices
    last = float(c.iloc[-1])
    def ret(n): return (last / float(c.iloc[-n - 1]) - 1) * 100 if len(c) > n else float("nan")
    sma50, sma200 = float(c.tail(50).mean()), float(c.tail(200).mean())
    hi52, lo52 = float(c.tail(252).max()), float(c.tail(252).min())
    vol = float((c.pct_change().tail(20).std()) * (252 ** 0.5) * 100)
    return (
        f"Last close {last:,.4f}. "
        f"Returns: 1w {ret(5):+.1f}%, 1m {ret(21):+.1f}%, 3m {ret(63):+.1f}%. "
        f"vs 50d SMA {((last/sma50-1)*100):+.1f}%, vs 200d SMA {((last/sma200-1)*100):+.1f}%. "
        f"52w range {lo52:,.2f}–{hi52:,.2f} (now { (last-lo52)/(hi52-lo52)*100:.0f}% of range). "
        f"Annualized 20d vol {vol:.0f}%."
    )


# ----------------------------- prompt -------------------------------
def build_prompt(symbol):
    m = INSTRUMENTS[symbol]
    kind = "single-stock CFD" if m.get("equity") else "macro instrument"
    return f"""You are a macro/market strategist producing a short-term directional view for a systematic trader.

Instrument: {m['name']} ({symbol}) — a {kind}.
Primary drivers: {m['drivers']}.
Recent price action: {price_context(symbol)}
Today: {_today()}.

You HAVE web search — use it and base your view on what you actually retrieve. Do NOT claim the search
quota is exhausted or that you cannot verify catalysts; issue the searches and read the results.
Find the LATEST market-moving developments for this instrument over the past ~1-2 weeks
(central-bank communications and expectations, economic-data surprises vs consensus, positioning, geopolitics,
and for equities: earnings/guidance/company news). Weigh what is NEW versus already priced in.

Then output a directional tilt for roughly the next 5-10 trading days. Be honest: if there is no clear edge,
say so with a tilt near 0 and low confidence. Do not anchor to the price trend alone.

End your reply with EXACTLY one fenced JSON block, no text after it:
```json
{{"stance": "long|short|flat", "tilt": <float -1..1>, "confidence": <float 0..1>,
 "horizon_days": <int>, "rationale": "<2-3 sentences citing the specific current catalysts>",
 "key_drivers": ["<short phrase>", "..."]}}
```"""


# --------------------------- Claude call ----------------------------
def _extract_json(text):
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
    if not m:
        raise ValueError("no JSON block in response")
    return json.loads(m.group(1))


def _call_claude(symbol):
    import anthropic
    # Talk to Anthropic DIRECTLY — bypass any local ANTHROPIC_BASE_URL proxy
    # (e.g. headroom), which injects tools this standalone script can't fulfill.
    # Cost rules learned the hard way (2026-07-09, $20 burned):
    #  - timeout=240: a slow web-search turn is BILLED even if we abandon it, so
    #    give it room to finish; the workflow's step timeout bounds the run.
    #  - max_retries=0: an SDK retry after a timeout can double-bill (the first
    #    request may still complete server-side). Retrying is the runner's job,
    #    and thanks to the on-disk cache a retried symbol re-pays nothing once done.
    client = anthropic.Anthropic(base_url="https://api.anthropic.com",
                                 timeout=240.0, max_retries=0)
    messages = [{"role": "user", "content": build_prompt(symbol)}]
    text_out, results, errors = "", 0, 0
    search_ids = set()          # dedupe by block id — continuations can repeat blocks
    in_tok = out_tok = cache_w = cache_r = 0
    for _ in range(2):  # cost cap: initial + ONE continuation (searches re-bill per round)
        try:
            # STREAM, don't block: on a non-streaming call the client can time out
            # and abandon a request the server still completes AND BILLS (learned
            # 2026-07-10: five timed-out calls = invisible spend). A stream keeps
            # bytes flowing, so slow-but-healthy turns finish instead of being paid
            # for and thrown away. timeout=240 still guards a truly dead stream.
            with client.messages.stream(
                model=MODEL, max_tokens=2500, tools=[WEB_SEARCH], messages=messages,
            ) as s:
                resp = s.get_final_message()
        except anthropic.BadRequestError as e:
            if "credit balance is too low" in str(e).lower():
                raise CreditsExhausted("Anthropic credits exhausted") from e
            raise
        u = getattr(resp, "usage", None)
        if u:
            in_tok += getattr(u, "input_tokens", 0) or 0
            out_tok += getattr(u, "output_tokens", 0) or 0
            cache_w += getattr(u, "cache_creation_input_tokens", 0) or 0
            cache_r += getattr(u, "cache_read_input_tokens", 0) or 0
        for b in resp.content:
            if b.type == "server_tool_use":
                search_ids.add(getattr(b, "id", len(search_ids)))
            elif b.type == "web_search_tool_result":
                if isinstance(b.content, list):
                    results += len(b.content)
                else:                                  # error object (max_uses_exceeded, etc.)
                    errors += 1
        text_out = "".join(b.text for b in resp.content if b.type == "text")
        if resp.stop_reason == "pause_turn":
            # continuation re-reads this whole prefix — mark it cacheable so the
            # re-read bills at 10% instead of full price. This is what makes a
            # heavy-search day cost close to a normal one.
            blocks = [b.model_dump(exclude_none=True) for b in resp.content]
            if blocks:
                blocks[-1]["cache_control"] = {"type": "ephemeral"}
            messages.append({"role": "assistant", "content": blocks})
            continue
        break
    return (_extract_json(text_out), len(search_ids), results, errors,
            in_tok, out_tok, cache_w, cache_r)


def get_tilt(symbol, refresh=False, verbose=False):
    """Return a validated Tilt for `symbol`, cached per (symbol, day)."""
    if symbol not in INSTRUMENTS:
        raise KeyError(f"unknown symbol {symbol}; known: {list(INSTRUMENTS)}")
    key = os.path.join(CACHE, f"{symbol}_{_today()}.json")
    if not refresh and os.path.exists(key):
        d = json.load(open(key)); d["source"] = "cache"
        return Tilt(**d)

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set — export it to call Claude.")

    (raw, searches, results, errors,
     in_tok, out_tok, cache_w, cache_r) = _call_claude(symbol)
    tilt = Tilt(
        symbol=symbol,
        stance=str(raw.get("stance", "flat")).lower(),
        tilt=max(-1.0, min(1.0, float(raw.get("tilt", 0.0)))),
        confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.0)))),
        horizon_days=int(raw.get("horizon_days", 7)),
        rationale=str(raw.get("rationale", ""))[:600],
        key_drivers=list(raw.get("key_drivers", []))[:6],
        as_of=_today(), model=MODEL, source="claude+websearch",
        grounded=(results >= 3 and errors == 0), searches=searches,
        in_tokens=in_tok, out_tokens=out_tok,
        cache_w_tokens=cache_w, cache_r_tokens=cache_r,
    )
    json.dump(asdict(tilt), open(key, "w"))
    with _log_lock, open(LOG, "a") as fh:
        fh.write(json.dumps({**asdict(tilt), "ts": time.time()}) + "\n")
    if verbose:
        flag = f"{searches} searches" if tilt.grounded else "NOT GROUNDED"
        print(f"  {symbol:<7} {tilt.stance:<5} tilt {tilt.tilt:+.2f} conf {tilt.confidence:.2f} "
              f"({tilt.horizon_days}d, {flag}): {tilt.rationale[:80]}")
    return tilt


if __name__ == "__main__":
    import sys
    syms = sys.argv[1:] or ["XAUUSD"]
    for s in syms:
        print(get_tilt(s, verbose=True))
