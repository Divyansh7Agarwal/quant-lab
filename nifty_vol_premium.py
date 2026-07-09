"""NIFTY volatility-risk-premium backtest v2 — corrected specification.

v1 had two modeling errors (not findings): fixed-percent strikes (ignores vol level;
no practitioner does this) and 100%-of-capital-per-cycle compounding (one max loss
= ruin by construction). v2 uses the standard practice, chosen a priori:
  * strikes by STANDARD DEVIATION: short call/put at +/-1.0 SD, wings at +/-1.6 SD
    (SD = IV * sqrt(T)), so strikes widen automatically when vol is high;
  * sizing: each cycle risks 20% of equity (fractional Kelly-ish, survivable).

Everything else unchanged: monthly cycles, BS with IV = India VIX (flat vol — real
chains have skew, favorable to put sellers, so income is understated), settle on
actual NIFTY, 10% premium haircut for friction, full history including 2008.
Report full period AND 2016+ (India VIX early data is thin; market structure
modernized) — BOTH shown, no cherry-picking.
"""
import os, sys, math
import numpy as np
import pandas as pd

CYCLE = 21
COST_HAIRCUT = 0.10
SD_SHORT = 1.0
SD_WING = 1.6
RISK_FRAC = 0.20      # fraction of equity at risk per cycle


def bs(S, K, T, sig, kind):
    if T <= 0 or sig <= 0:
        return max(0.0, (S - K) if kind == "c" else (K - S))
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    if kind == "c":
        return S * N(d1) - K * N(d2)
    return K * N(-d2) - S * N(-d1)


def fetch():
    import yfinance as yf
    def get(t):
        h = yf.download(t, period="max", interval="1d", auto_adjust=False, progress=False)["Close"]
        return h.squeeze().dropna()
    return pd.DataFrame({"S": get("^NSEI"), "iv": get("^INDIAVIX")}).dropna()


def run(df):
    idx = df.index
    rows = []
    for i in range(0, len(df) - CYCLE, CYCLE):
        S0, ST = float(df["S"].iloc[i]), float(df["S"].iloc[i + CYCLE])
        sig = float(df["iv"].iloc[i]) / 100
        T = 30 / 365
        sd = sig * math.sqrt(T)

        kc, kp = S0 * math.exp(SD_SHORT * sd), S0 * math.exp(-SD_SHORT * sd)
        kcw, kpw = S0 * math.exp(SD_WING * sd), S0 * math.exp(-SD_WING * sd)

        credit = (bs(S0, kc, T, sig, "c") + bs(S0, kp, T, sig, "p")
                  - bs(S0, kcw, T, sig, "c") - bs(S0, kpw, T, sig, "p")) * (1 - COST_HAIRCUT)
        payoff = (max(0, ST - kc) - max(0, ST - kcw)
                  + max(0, kp - ST) - max(0, kpw - ST))
        max_loss = max(kcw - kc, kp - kpw) - credit
        ret = (credit - payoff) / max_loss

        rows.append(dict(date=idx[i], S0=S0, ST=ST, iv=sig * 100,
                         move=(ST / S0 - 1) * 100, ret=ret,
                         credit_pct=credit / S0 * 100))
    return pd.DataFrame(rows).set_index("date")


def stats(r, risk_frac=RISK_FRAC, cpy=12):
    r = r.dropna()
    eq = (1 + risk_frac * r).cumprod()          # 20% of equity at risk per cycle
    yrs = len(r) / cpy
    ann = eq.iloc[-1] ** (1 / yrs) - 1
    port = risk_frac * r
    sh = port.mean() / port.std() * math.sqrt(cpy) if port.std() > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    return dict(ann=ann, sharpe=sh, maxdd=dd, hit=(r > 0).mean(),
                avg=r.mean(), worst=r.min())


def report(res, label):
    s = stats(res["ret"])
    print(f"  {label:<18} ann {s['ann']*100:+6.1f}%  sharpe {s['sharpe']:5.2f}  "
          f"maxDD {s['maxdd']*100:6.1f}%  hit {s['hit']*100:3.0f}%  "
          f"avg/cycle {s['avg']*100:+5.1f}%  worst {s['worst']*100:+6.1f}%")


def main():
    print("Fetching NIFTY + India VIX (max history)...")
    df = fetch()
    print(f"{len(df)} days: {df.index[0].date()} -> {df.index[-1].date()}")

    res = run(df)
    print(f"{len(res)} cycles | avg entry IV {res['iv'].mean():.1f} | "
          f"avg credit {res['credit_pct'].mean():.2f}% of spot | "
          f"1SD-condor, 20% equity risk/cycle, 10% cost haircut\n")

    report(res, "FULL 2008-2026")
    report(res[res.index >= "2016-01-01"], "2016-2026")
    report(res[res.index >= "2021-01-01"], "2021-2026")

    # NIFTY benchmark (fully invested)
    nif = res["ST"] / res["S0"] - 1
    eq = (1 + nif).cumprod()
    yrs = len(nif) / 12
    print(f"\n  NIFTY buy&hold     ann {eq.iloc[-1]**(1/yrs)-1:+.1%}  "
          f"(fully invested, for scale)")

    print("\n  5 worst condor cycles:")
    for d, w in res.nsmallest(5, "ret").iterrows():
        print(f"    {d.date()}  ret {w['ret']*100:+6.1f}%  NIFTY {w['move']:+5.1f}%  IV {w['iv']:.0f}")

    med = res["iv"].median()
    hi, lo = res[res["iv"] > med]["ret"], res[res["iv"] <= med]["ret"]
    print(f"\n  by entry IV (median {med:.1f}):")
    print(f"    IV high: avg/cycle {hi.mean()*100:+5.1f}%  hit {(hi>0).mean()*100:.0f}%")
    print(f"    IV low : avg/cycle {lo.mean()*100:+5.1f}%  hit {(lo>0).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
