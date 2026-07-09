"""LLM news/sentiment signal — the candidate *alpha* layer.

Pulls recent headlines per ticker, scores each in [-1, 1], aggregates into one
recency-weighted sentiment score per asset. This plugs into the engine as just
another signal with a small weight.

Two scoring backends:
  * Claude  — used automatically if ANTHROPIC_API_KEY is set (cheap haiku model).
  * Heuristic lexicon — transparent fallback so it runs today with no key/cost.

HONEST LIMITATION: free news APIs only return *recent* items, so this signal can't
be backtested over years. Its validation path is FORWARD paper-trading, not a
historical backtest. Scores are cached to disk so you don't re-pay for re-runs.
"""
import os, json, math, hashlib
import datetime as dt
import yfinance as yf

CACHE = os.path.join(os.path.dirname(__file__), "cache", "sentiment")
os.makedirs(CACHE, exist_ok=True)

HAIKU = "claude-haiku-4-5-20251001"   # cheap + fast: right tool for scoring headlines

_POS = {"beat", "beats", "surge", "surges", "record", "upgrade", "upgraded", "growth",
        "profit", "soar", "soars", "rally", "rallies", "strong", "gain", "gains",
        "raises", "raised", "outperform", "bullish", "tops", "jump", "jumps", "wins",
        "approval", "expand", "expands", "high", "highs", "boost", "rebound"}
_NEG = {"miss", "misses", "plunge", "plunges", "cut", "cuts", "downgrade", "downgraded",
        "lawsuit", "probe", "fall", "falls", "weak", "loss", "losses", "bankruptcy",
        "slump", "bearish", "warn", "warns", "recall", "layoff", "layoffs", "drop",
        "drops", "sink", "sinks", "fraud", "halt", "delay", "delays", "low", "lows"}


# --------------------------- news fetch ---------------------------
def get_headlines(ticker, max_age_days=5, limit=12):
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out, now = [], dt.datetime.now(dt.timezone.utc)
    for it in items:
        c = it.get("content", it)
        title = c.get("title") or ""
        pub = c.get("pubDate") or c.get("displayTime")
        if not title:
            continue
        age_days = 0.0
        if isinstance(pub, str):
            try:
                t = dt.datetime.fromisoformat(pub.replace("Z", "+00:00"))
                age_days = (now - t).total_seconds() / 86400
            except Exception:
                pass
        if age_days <= max_age_days:
            out.append({"title": title.strip(),
                        "summary": (c.get("summary") or c.get("description") or "")[:280],
                        "age_days": round(age_days, 2)})
        if len(out) >= limit:
            break
    return out


# --------------------------- scoring ------------------------------
def _heuristic_score(text):
    toks = [w.strip(".,!?:;\"'()").lower() for w in text.split()]
    p = sum(t in _POS for t in toks)
    n = sum(t in _NEG for t in toks)
    return 0.0 if p + n == 0 else (p - n) / (p + n)


def _claude_scores(headlines):
    import anthropic
    client = anthropic.Anthropic()
    lines = "\n".join(f"{i+1}. {h['title']}" for i, h in enumerate(headlines))
    prompt = (
        "Score each headline's likely short-term impact on the company's stock, "
        "from -1.0 (very bearish) to 1.0 (very bullish), 0 if neutral/irrelevant.\n"
        "Return ONLY a JSON array of numbers, same order, no prose.\n\n" + lines
    )
    msg = client.messages.create(
        model=HAIKU, max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    txt = msg.content[0].text.strip()
    txt = txt[txt.find("["): txt.rfind("]") + 1]
    arr = json.loads(txt)
    return [max(-1.0, min(1.0, float(x))) for x in arr]


def _cache_key(ticker, headlines):
    h = hashlib.md5((ticker + "|".join(x["title"] for x in headlines)).encode()).hexdigest()[:16]
    return os.path.join(CACHE, f"{ticker}_{h}.json")


def sentiment_signal(ticker, use_claude=None, verbose=False):
    """Return (score in [-1,1], n_headlines). Recency-weighted average of per-headline
    scores. Cached by (ticker, headline set)."""
    headlines = get_headlines(ticker)
    if not headlines:
        return 0.0, 0
    key = _cache_key(ticker, headlines)
    if os.path.exists(key):
        return tuple(json.load(open(key)))

    use_claude = (os.getenv("ANTHROPIC_API_KEY") is not None) if use_claude is None else use_claude
    try:
        scores = _claude_scores(headlines) if use_claude else \
                 [_heuristic_score(h["title"] + " " + h["summary"]) for h in headlines]
        backend = "claude" if use_claude else "heuristic"
    except Exception as e:                       # noqa: BLE001
        if verbose:
            print(f"   claude failed ({e}); using heuristic")
        scores = [_heuristic_score(h["title"] + " " + h["summary"]) for h in headlines]
        backend = "heuristic"

    # recency weight: newer headlines count more (half-life ~2 days)
    wsum = score = 0.0
    for h, s in zip(headlines, scores):
        w = math.exp(-h["age_days"] / 2.0)
        score += w * s
        wsum += w
    final = round(score / wsum, 3) if wsum else 0.0
    json.dump([final, len(headlines)], open(key, "w"))
    if verbose:
        print(f"   {ticker}: {backend} sentiment {final:+.2f} from {len(headlines)} headlines")
    return final, len(headlines)


if __name__ == "__main__":
    for tk in ["AAPL", "NVDA", "TSLA", "SPY"]:
        s, n = sentiment_signal(tk, verbose=True)
