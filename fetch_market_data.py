"""
fetch_market_data.py
Single source of truth for the Current PF - Weekly dashboard's live feed.
For every stock in TICKERS, computes:
  - CMP (current market price)
  - Weekly Heikin-Ashi candle-streak label for the last 4 weeks
    ("1st white" / "2nd white" / "1st red" / "2nd red"), matching the
    Chartink weekly Heikin-Ashi chart color logic:
        first white after a red streak  -> "1st white" (blue)
        white continuing a white streak -> "2nd white" (green)
        first red after a white streak  -> "1st red"   (light red)
        red continuing a red streak     -> "2nd red"   (dark red)
  - Weekly StochRSI(14) (the same value shown on the Chartink chart's
    StochRSI(14) panel)

All three come from the same weekly OHLC series pulled via yfinance, so
there's no need to open Chartink manually every week.

Writes market_data.json:
{
  "asof": "2026-07-14T15:30:00+05:30",
  "market_open": true,
  "prices":   {"CUPID": 217.4, ...},
  "trend":    {"CUPID": {"week1": "1st white", "week2": "2nd white", "week3": "2nd white", "week4": "2nd white"}, ...},
  "stochrsi": {"CUPID": 100.0, ...},
  "errors":   ["SYMBOL:price", "SYMBOL:trend", ...]
}

NOTE: the most recent week's candle is still "live" until the trading
week closes, so week4 can shift color during the week — same as it
would on Chartink's own weekly chart before Friday close.
"""

import json
import time
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

# --- Keep this in sync with the dashboard's stock list ---
TICKERS = [
    "CUPID", "COFORGE", "SUBEXLTD", "ETERNAL", "UJJIVANSFB",
    "JSWINFRA", "MOSCHIP", "STALLION", "EXIDEIND", "EMCURE",
    "JAMNAAUTO", "CHENNPETRO", "TATACAP", "PREMIERENE", "CHAMBLFERT",
    "RITES", "GPPL", "JSWENERGY", "WIPRO", "TEJASNET", "SARDAEN",
]

IST = pytz.timezone("Asia/Kolkata")
WEEKLY_LOOKBACK = "3y"  # matches your Chartink chart's 3-year weekly period


def to_yf_symbol(nse_symbol: str) -> str:
    return f"{nse_symbol}.NS"


def is_market_open(now_ist) -> bool:
    if now_ist.weekday() >= 5:
        return False
    return dtime(9, 15) <= now_ist.time() <= dtime(15, 30)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha_close.iloc[i - 1]) / 2)
    return pd.DataFrame({"open": ha_open, "close": ha_close.values}, index=df.index)


def candle_streak_labels(ha: pd.DataFrame, n: int = 4):
    """Label every HA candle as '1st'/'2nd' white/red based on streak continuation,
    computed over full history so streaks carry over correctly, then return the
    last n labels (week1 = oldest of the 4, week4 = most recent)."""
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


def fetch_current_price(symbol: str):
    try:
        tk = yf.Ticker(symbol)
        fi = tk.fast_info
        price = fi.get("last_price") if hasattr(fi, "get") else getattr(fi, "last_price", None)
        if price is None:
            hist = tk.history(period="1d", interval="1m")
            if hist.empty:
                hist = tk.history(period="5d")
            price = float(hist["Close"].iloc[-1])
        return round(float(price), 2)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] price {symbol}: {exc}")
        return None


def fetch_weekly_analysis(symbol: str):
    """Returns (week_labels_dict, latest_stochrsi) or (None, None) on failure."""
    try:
        hist = yf.Ticker(symbol).history(period=WEEKLY_LOOKBACK, interval="1wk")
        hist = hist.dropna(subset=["Open", "High", "Low", "Close"])
        if len(hist) < 20:
            raise ValueError(f"not enough weekly history ({len(hist)} bars)")

        ha = heikin_ashi(hist)
        trend_labels = candle_streak_labels(ha, n=4)
        weeks = {
            "week1": trend_labels[0],
            "week2": trend_labels[1],
            "week3": trend_labels[2],
            "week4": trend_labels[3],
        }

        srsi_series = stoch_rsi(hist["Close"], period=14)
        latest_srsi = srsi_series.iloc[-1]
        latest_srsi = None if pd.isna(latest_srsi) else round(float(latest_srsi), 1)

        return weeks, latest_srsi
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] weekly analysis {symbol}: {exc}")
        return None, None


def main():
    now_ist = datetime.now(IST)
    prices, trend, stochrsi, errors = {}, {}, {}, []

    for symbol in TICKERS:
        yf_symbol = to_yf_symbol(symbol)

        price = fetch_current_price(yf_symbol)
        if price is not None:
            prices[symbol] = price
        else:
            errors.append(f"{symbol}:price")

        weeks, srsi_val = fetch_weekly_analysis(yf_symbol)
        if weeks is not None:
            trend[symbol] = weeks
        else:
            errors.append(f"{symbol}:trend")
        if srsi_val is not None:
            stochrsi[symbol] = srsi_val
        else:
            errors.append(f"{symbol}:stochrsi")

        time.sleep(0.3)  # be polite to the endpoint

    output = {
        "asof": now_ist.isoformat(),
        "market_open": is_market_open(now_ist),
        "prices": prices,
        "trend": trend,
        "stochrsi": stochrsi,
        "errors": errors,
    }

    with open("market_data.json", "w") as f:
        json.dump(output, f, indent=2)

    print(
        f"Prices {len(prices)}/{len(TICKERS)} · Trend {len(trend)}/{len(TICKERS)} "
        f"· StochRSI {len(stochrsi)}/{len(TICKERS)} @ {now_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )
    if errors:
        print(f"Issues: {errors}")


if __name__ == "__main__":
    main()
