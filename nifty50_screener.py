"""
nifty50_screener.py
Weekly screener across the Nifty 50 universe (much smaller/faster than the
Nifty 500 version — good if the 500-stock scan is timing out or unreliable).
Runs once a week (Friday, after market close) via update_nifty50_screener.yml.

For every Nifty 50 stock, computes:
  - CMP (last traded/close price)
  - Weekly Heikin-Ashi candle-streak label for the last 4 weeks
  - Weekly StochRSI(14)

...then tags each stock with a signal:
  - "buy"     : current StochRSI > 50
  - "sell"    : StochRSI was >=95 last week (near-100/overbought) and has
                now dropped to <=80 this week
  - "neutral" : anything else

Writes nifty50_screener.json:
{
  "asof": "...",
  "universe_size": 50,
  "stocks": {
    "RELIANCE": {"cmp": 3100.5, "week1":"...", ..., "week4":"...",
                 "stochrsi": 82.3, "stochrsi_prev": 91.0, "signal": "buy"},
    ...
  },
  "errors": ["SYMBOL", ...]
}

METHOD: fetches each ticker individually (yf.Ticker(...).history()) — the
same proven approach validated against Chartink for the portfolio tool.
"""

import io
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

IST = pytz.timezone("Asia/Kolkata")
WEEKLY_LOOKBACK = "3y"

NSE_50_CSV_URL = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
NSE_HEADERS = {
    # NSE blocks requests without a browser-like User-Agent
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,application/csv,*/*",
}


def get_nifty50_symbols():
    """Download and parse the official Nifty 50 constituent list from NSE."""
    resp = requests.get(NSE_50_CSV_URL, headers=NSE_HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    symbol_col = next(c for c in df.columns if c.strip().lower() == "symbol")
    return sorted(df[symbol_col].dropna().unique().tolist())


def to_yf_symbol(nse_symbol: str) -> str:
    return f"{nse_symbol}.NS"


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha_close.iloc[i - 1]) / 2)
    return pd.DataFrame({"open": ha_open, "close": ha_close.values}, index=df.index)


def candle_streak_labels(ha: pd.DataFrame, n: int = 4):
    colors = ["white" if c >= o else "red" for o, c in zip(ha["open"], ha["close"])]
    labels, prev = [], None
    for color in colors:
        labels.append(f"2nd {color}" if color == prev else f"1st {color}")
        prev = color
    return labels[-n:]


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # NOTE: do NOT replace 0 with NaN before dividing — that was the bug.
    # A clean multi-week run (no losing weeks at all) makes avg_loss exactly 0,
    # which should correctly yield RSI=100 (division by 0 -> inf -> RSI 100),
    # not NaN. We only need to special-case the true 0/0 (no movement at all).
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    r = 100 - (100 / (1 + rs))
    r = r.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    r = r.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return r


def stoch_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    r = rsi(series, period)
    lowest = r.rolling(period).min()
    highest = r.rolling(period).max()
    with np.errstate(divide="ignore", invalid="ignore"):
        k = ((r - lowest) / (highest - lowest)) * 100
    # RSI flat over the whole window (e.g. pinned at 100 during a clean
    # multi-week rally, or pinned at 0 during a clean sell-off) makes the
    # range 0 -> undefined ratio. In that case the RSI's own value already
    # tells us how extreme things are, so use it directly instead of
    # defaulting to a meaningless "neutral 50".
    k = k.mask(highest == lowest, r)
    return k


def signal_for(stochrsi_now, stochrsi_prev):
    """
    New rule (per your criteria):
      - "sell"    : StochRSI was near the top (>=95, i.e. was ~100/overbought)
                    last week and has now come down to <=80 this week
                    -> momentum rolling over from overbought
      - "buy"     : current StochRSI > 50 (upside bias)
      - "neutral" : anything else
    """
    if stochrsi_prev is not None and stochrsi_now is not None:
        if stochrsi_prev >= 95 and stochrsi_now <= 80:
            return "sell"
    if stochrsi_now is not None and stochrsi_now > 50:
        return "buy"
    return "neutral"


def analyze_one(symbol: str, hist: pd.DataFrame):
    hist = hist.dropna(subset=["Open", "High", "Low", "Close"])
    if len(hist) < 20:
        raise ValueError(f"not enough weekly history ({len(hist)} bars)")
    ha = heikin_ashi(hist)
    labels = candle_streak_labels(ha, n=4)
    srsi_series = stoch_rsi(hist["Close"], period=14)

    def _clean(v):
        return None if pd.isna(v) else round(float(v), 1)

    latest_srsi = _clean(srsi_series.iloc[-1])
    prev_srsi = _clean(srsi_series.iloc[-2]) if len(srsi_series) >= 2 else None

    cmp_price = round(float(hist["Close"].iloc[-1]), 2)
    week4 = labels[3]
    return {
        "cmp": cmp_price,
        "week1": labels[0], "week2": labels[1], "week3": labels[2], "week4": week4,
        "stochrsi": latest_srsi,
        "stochrsi_prev": prev_srsi,
        "signal": signal_for(latest_srsi, prev_srsi),
    }


def fetch_one(nse_symbol: str):
    """Fetch weekly history for a single stock and analyze it. Sequential,
    proven-correct method (same approach validated against Chartink for
    the portfolio tool) — slower than batch downloading, but reliable."""
    yf_symbol = to_yf_symbol(nse_symbol)
    hist = yf.Ticker(yf_symbol).history(period=WEEKLY_LOOKBACK, interval="1wk", auto_adjust=False)
    return analyze_one(nse_symbol, hist)


def main():
    now_ist = datetime.now(IST)
    symbols = get_nifty50_symbols()
    print(f"Fetched {len(symbols)} Nifty 50 symbols from NSE")

    results, errors = {}, []
    for i, symbol in enumerate(symbols, 1):
        try:
            results[symbol] = fetch_one(symbol)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {symbol}: {exc}")
            errors.append(symbol)
        time.sleep(0.3)  # be polite to the endpoint

    output = {
        "asof": now_ist.isoformat(),
        "universe_size": len(symbols),
        "stocks": results,
        "errors": errors,
    }

    with open("nifty50_screener.json", "w") as f:
        json.dump(output, f, indent=2)

    buy = sum(1 for v in results.values() if v["signal"] == "buy")
    sell = sum(1 for v in results.values() if v["signal"] == "sell")
    print(f"Analyzed {len(results)}/{len(symbols)} · buy={buy} · sell={sell} · errors={len(errors)}")


if __name__ == "__main__":
    main()
