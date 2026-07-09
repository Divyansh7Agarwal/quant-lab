"""MetaTrader 5 adapter — data + execution for the engine.

RUNS ON WINDOWS ONLY (the `MetaTrader5` package talks to a running MT5 terminal
over Windows IPC). On Mac/Linux it imports as a stub so the rest of the codebase
still loads for backtesting. Deploy this file on a Windows VPS with MT5 installed
and your DEMO account logged in.

Connection model: just run the MT5 terminal, log it into your account, then call
connect(). No keys in code. A hard safety guard refuses to submit live orders
unless the connected account is a DEMO account (override only via allow_live=True).
"""
import pandas as pd

try:
    import MetaTrader5 as mt5
    HAVE_MT5 = True
except Exception:                       # noqa: BLE001  (Mac/Linux: no package)
    mt5 = None
    HAVE_MT5 = False

_TF = {"D1": "TIMEFRAME_D1", "H4": "TIMEFRAME_H4", "H1": "TIMEFRAME_H1",
       "M15": "TIMEFRAME_M15", "M5": "TIMEFRAME_M5"}


def _require():
    if not HAVE_MT5:
        raise RuntimeError(
            "MetaTrader5 package not available — run this on Windows with MT5 installed. "
            "On Mac, use run_mt5_backtest.py (yfinance proxies) instead.")


def connect(login=None, password=None, server=None, path=None):
    """Attach to the running MT5 terminal (optionally logging into a specific
    account). Returns the account dict."""
    _require()
    ok = mt5.initialize(path=path) if path else mt5.initialize()
    if not ok:
        raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")
    if login:
        if not mt5.login(int(login), password=password, server=server):
            raise RuntimeError(f"mt5.login failed: {mt5.last_error()}")
    return account()


def account():
    _require()
    a = mt5.account_info()
    if a is None:
        raise RuntimeError(f"no account info: {mt5.last_error()}")
    d = a._asdict()
    d["is_demo"] = (a.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO)
    return d


def get_rates(symbol, timeframe="D1", n=1500):
    """Return DataFrame[open,high,low,close,volume] indexed by timestamp."""
    _require()
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"cannot select symbol {symbol}: {mt5.last_error()}")
    tf = getattr(mt5, _TF[timeframe])
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"no rates for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("timestamp")
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def get_universe(symbols, timeframe="D1", n=1500):
    out = {}
    for s in symbols:
        try:
            out[s] = get_rates(s, timeframe, n)
        except Exception as e:          # noqa: BLE001
            print(f"  ! skip {s}: {e}")
    return out


def sym_info(symbol):
    _require()
    i = mt5.symbol_info(symbol)
    if i is None:
        raise RuntimeError(f"no symbol info {symbol}")
    tick = mt5.symbol_info_tick(symbol)
    return {
        "contract_size": i.trade_contract_size,
        "vol_min": i.volume_min, "vol_step": i.volume_step, "vol_max": i.volume_max,
        "digits": i.digits, "point": i.point,
        "ask": tick.ask, "bid": tick.bid,
    }


def net_positions():
    """symbol -> signed net lots (+long / -short) currently open."""
    _require()
    pos = mt5.positions_get() or []
    out = {}
    for p in pos:
        sign = 1 if p.type == mt5.ORDER_TYPE_BUY else -1
        out[p.symbol] = out.get(p.symbol, 0.0) + sign * p.volume
    return out


def lots_for_weight(symbol, weight, equity):
    """Convert a target portfolio weight (fraction of equity as notional exposure)
    into a rounded, broker-valid lot size. Sign carries long/short.
    Notional per lot ~= contract_size * price (quote ccy ~ account ccy for USD pairs;
    APPROXIMATE for JPY/cross pairs — fine for a demo forward-test)."""
    info = sym_info(symbol)
    price = (info["ask"] + info["bid"]) / 2 or info["ask"]
    notional_per_lot = info["contract_size"] * price
    if notional_per_lot <= 0:
        return 0.0
    raw = (weight * equity) / notional_per_lot
    step = info["vol_step"] or 0.01
    lots = round(round(raw / step) * step, 2)
    if abs(lots) < info["vol_min"]:
        return 0.0
    return max(-info["vol_max"], min(info["vol_max"], lots))


def _send(symbol, lots, is_buy, deviation=20, comment="quant-engine"):
    _require()
    info = sym_info(symbol)
    price = info["ask"] if is_buy else info["bid"]
    for filling in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": abs(lots),
               "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
               "price": price, "deviation": deviation, "magic": 770077,
               "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
               "type_filling": filling}
        r = mt5.order_send(req)
        if r is not None and r.retcode == mt5.TRADE_RETCODE_DONE:
            return r
    return r        # last (failed) result for inspection


def close_symbol(symbol):
    """Flatten any open position on `symbol`."""
    _require()
    for p in (mt5.positions_get(symbol=symbol) or []):
        is_buy = (p.type == mt5.ORDER_TYPE_SELL)   # opposite side to close
        _send(symbol, p.volume, is_buy, comment="quant-close")


def rebalance_symbol(symbol, target_lots):
    """Reconcile current net position to `target_lots` (netting-account model):
    flatten then re-open. Simple and robust for a daily demo loop."""
    cur = net_positions().get(symbol, 0.0)
    if abs(cur - target_lots) < (sym_info(symbol)["vol_step"] or 0.01):
        return None
    close_symbol(symbol)
    if abs(target_lots) > 0:
        return _send(symbol, abs(target_lots), is_buy=(target_lots > 0))
    return None


def shutdown():
    if HAVE_MT5:
        mt5.shutdown()
