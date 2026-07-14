"""
nifty500_screener.py
Weekly screener across the full Nifty 500 universe. Runs once a week
(Friday, after market close) via update_nifty500_screener.yml.

For every Nifty 500 stock, computes the same three things as the portfolio
tool (fetch_market_data.py):
  - CMP (last traded/close price)
  - Weekly Heikin-Ashi candle-streak label for the last 4 weeks
  - Weekly StochRSI(14)

...then buckets each stock into a signal category based on week4 + StochRSI:
  - fresh_buy    : week4 == "1st white"  (just turned up)
  - fresh_sell   : week4 == "1st red"    (just turned down)
  - continuing_up   : week4 == "2nd white" (uptrend continuing)
  - continuing_down : week4 == "2nd red"   (downtrend continuing)

Writes nifty500_screener.json:
{
  "asof": "...",
  "universe_size": 500,
  "stocks": {
    "RELIANCE": {"cmp": 3100.5, "week1":"...", "week2":"...", "week3":"...",
                 "week4":"...", "stochrsi": 82.3, "signal": "continuing_up"},
    ...
  },
  "errors": ["SYMBOL", ...]   // symbols that failed to fetch entirely
}

NOTE ON SPEED: fetching 500 tickers one-by-one is slow and easy to get
rate-limited, so this pulls weekly OHLC in bulk via yf.download() in
chunks, rather than yf.Ticker(...).history() per symbol.
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
CHUNK_SIZE = 40          # tickers per yf.download() batch
CHUNK_PAUSE_SEC = 2      # pause between batches to be polite

NSE_500_CSV_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NSE_HEADERS = {
    # NSE blocks requests without a browser-like User-Agent
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,application/csv,*/*",
}


def get_nifty500_symbols():
    """Download and parse the official Nifty 500 constituent list from NSE."""
    resp = requests.get(NSE_500_CSV_URL, headers=NSE_HEADERS, timeout=30)
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
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stoch_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    r = rsi(series, period)
    lowest = r.rolling(period).min()
    highest = r.rolling(period).max()
    return ((r - lowest) / (highest - lowest).replace(0, np.nan)) * 100


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


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    now_ist = datetime.now(IST)
    symbols = get_nifty500_symbols()
    print(f"Fetched {len(symbols)} Nifty 500 symbols from NSE")

    results, errors = {}, []
    yf_to_nse = {to_yf_symbol(s): s for s in symbols}
    all_yf_symbols = list(yf_to_nse.keys())

    for batch in chunked(all_yf_symbols, CHUNK_SIZE):
        try:
            data = yf.download(
                tickers=" ".join(batch),
                period=WEEKLY_LOOKBACK,
                interval="1wk",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] batch download failed: {exc}")
            errors.extend(yf_to_nse[s] for s in batch)
            continue

        for yf_symbol in batch:
            nse_symbol = yf_to_nse[yf_symbol]
            try:
                if len(batch) == 1:
                    hist = data
                else:
                    hist = data[yf_symbol]
                results[nse_symbol] = analyze_one(nse_symbol, hist)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {nse_symbol}: {exc}")
                errors.append(nse_symbol)

        time.sleep(CHUNK_PAUSE_SEC)

    output = {
        "asof": now_ist.isoformat(),
        "universe_size": len(symbols),
        "stocks": results,
        "errors": errors,
    }

    with open("nifty500_screener.json", "w") as f:
        json.dump(output, f, indent=2)

    buy = sum(1 for v in results.values() if v["signal"] == "buy")
    sell = sum(1 for v in results.values() if v["signal"] == "sell")
    print(f"Analyzed {len(results)}/{len(symbols)} · buy={buy} · sell={sell} · errors={len(errors)}")


if __name__ == "__main__":
    main()
