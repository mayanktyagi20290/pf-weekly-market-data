"""
debug_stochrsi.py — run this locally (where you have internet access) to see
exactly how StochRSI is being computed for a stock, so we can compare
against what Chartink shows and find the real mismatch.

    pip install yfinance pandas numpy --break-system-packages
    python debug_stochrsi.py PIRAMALFIN

Prints:
  - last 20 weekly closes
  - RSI(14) series (Wilder's smoothing) - last 10 values
  - RSI(14) series (Cutler's / simple-average variant) - last 10 values
  - StochRSI computed from each RSI variant - last 10 values
  - the exact "current" and "previous" values our screener would use

Compare the bottom StochRSI numbers against the value Chartink shows on
its weekly StochRSI(14) panel for the same stock. Whichever variant
(Wilder vs Cutler) lines up with Chartink tells us which formula to use
everywhere.
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf


def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_cutler(series: pd.Series, period: int = 14) -> pd.Series:
    """Simple-moving-average based RSI (a common alternative to Wilder's)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stoch_rsi(rsi_series: pd.Series, period: int = 14, smooth_k: int = 1) -> pd.Series:
    lowest = rsi_series.rolling(period).min()
    highest = rsi_series.rolling(period).max()
    raw_k = ((rsi_series - lowest) / (highest - lowest).replace(0, np.nan)) * 100
    return raw_k.rolling(smooth_k).mean() if smooth_k > 1 else raw_k


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "PIRAMALFIN"
    yf_symbol = f"{symbol}.NS"

    print(f"\n=== {symbol} ({yf_symbol}) ===\n")

    hist = yf.Ticker(yf_symbol).history(period="3y", interval="1wk", auto_adjust=False)
    hist = hist.dropna(subset=["Open", "High", "Low", "Close"])
    print(f"Weekly bars fetched: {len(hist)}")
    print(f"Last 6 weekly closes (with dates):")
    print(hist["Close"].tail(6).to_string())

    close = hist["Close"]

    print("\n--- RSI(14) — Wilder's smoothing (what our screener currently uses) ---")
    r_wilder = rsi_wilder(close, 14)
    print(r_wilder.tail(6).round(2).to_string())

    print("\n--- RSI(14) — Cutler's / simple-average variant ---")
    r_cutler = rsi_cutler(close, 14)
    print(r_cutler.tail(6).round(2).to_string())

    print("\n--- StochRSI(14) from Wilder RSI, unsmoothed ---")
    srsi_wilder = stoch_rsi(r_wilder, 14, smooth_k=1)
    print(srsi_wilder.tail(6).round(2).to_string())

    print("\n--- StochRSI(14) from Wilder RSI, 3-period smoothed K ---")
    srsi_wilder_smooth = stoch_rsi(r_wilder, 14, smooth_k=3)
    print(srsi_wilder_smooth.tail(6).round(2).to_string())

    print("\n--- StochRSI(14) from Cutler RSI, unsmoothed ---")
    srsi_cutler = stoch_rsi(r_cutler, 14, smooth_k=1)
    print(srsi_cutler.tail(6).round(2).to_string())

    print("\n--- StochRSI(14) from Cutler RSI, 3-period smoothed K ---")
    srsi_cutler_smooth = stoch_rsi(r_cutler, 14, smooth_k=3)
    print(srsi_cutler_smooth.tail(6).round(2).to_string())

    print(f"\n>>> Our screener currently reports (Wilder, unsmoothed): {srsi_wilder.iloc[-1]:.1f}")
    print(">>> Compare all FOUR variants above against Chartink's weekly StochRSI(14) value.")
    print(">>> Whichever variant matches tells us the fix needed.\n")


if __name__ == "__main__":
    main()
