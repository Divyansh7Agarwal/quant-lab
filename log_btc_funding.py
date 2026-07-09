"""Daily BTC funding-extreme logger — the forward test for the capitulation signal.

Computes today's funding z-score (trailing 3d vs 90d, backward-looking only) and
appends one row/day to btc_funding_log.jsonl. A LONG trigger (z < -2) is the rare
capitulation-bounce event. Resilient across exchanges (cloud CI geo-blocks some);
never crashes the pipeline. No API key, no cost.
"""
import os, sys, json, time
import datetime as dt
import pandas as pd
import ccxt

sys.path.insert(0, os.path.dirname(__file__))
from notify import notify

LOG = os.path.join(os.path.dirname(__file__), "btc_funding_log.jsonl")
Z_ENTRY = 2.0
# tried in order — first reachable exchange with usable funding history wins
EXCHANGES = ["binance", "bybit", "kucoin", "okx", "gate", "bitget"]


def fetch_funding(days=300):
    since0 = ccxt.binance().parse8601(
        (dt.date.today() - dt.timedelta(days=days)).isoformat() + "T00:00:00Z")
    for xid in EXCHANGES:
        try:
            ex = getattr(ccxt, xid)({"enableRateLimit": True, "timeout": 20000,
                                     "options": {"defaultType": "swap"}})
            rows, cursor = [], since0
            while True:
                batch = ex.fetch_funding_rate_history("BTC/USDT:USDT", since=cursor, limit=200)
                if not batch:
                    break
                rows += batch
                cursor = batch[-1]["timestamp"] + 1
                if len(batch) < 200 or cursor >= ex.milliseconds():
                    break
                time.sleep(ex.rateLimit / 1000)
            if len(rows) > 120:
                s = pd.Series({pd.to_datetime(r["timestamp"], unit="ms"): float(r["fundingRate"])
                               for r in rows})
                return s[~s.index.duplicated()].sort_index(), xid
        except Exception as e:                    # noqa: BLE001
            print(f"  funding: {xid} unreachable ({str(e)[:50]})")
    return None, None


def main():
    f8h, src = fetch_funding()
    if f8h is None:
        print("funding-z: no exchange reachable today — skipping (non-fatal)")
        return
    fd = f8h.resample("1D").sum()
    f3 = fd.rolling(3).mean()
    z = float(((f3 - f3.rolling(90).mean()) / f3.rolling(90).std()).iloc[-1])
    trigger = "LONG" if z < -Z_ENTRY else ("SHORT_ZONE" if z > Z_ENTRY else "none")
    row = {"date": dt.date.today().isoformat(), "fund_1d": float(fd.iloc[-1]),
           "fund_3d": float(f3.iloc[-1]), "z": round(z, 3), "trigger": trigger,
           "source": src, "ts": time.time()}
    if os.path.exists(LOG):
        for line in open(LOG):
            if json.loads(line).get("date") == row["date"]:
                print(f"funding-z: already logged today (z={z:+.2f}, {trigger})")
                return
    with open(LOG, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"funding-z: {row['date']} z={z:+.2f} trigger={trigger} via {src}")
    if trigger == "LONG":
        notify("Quant Lab: BUY SIGNAL",
               f"BTC crowd-panic fired (z={z:+.2f}). Historically a bounce follows. Open the dashboard.")


if __name__ == "__main__":
    main()
