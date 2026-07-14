"""
test_cupid.py — quick standalone check for one stock, before deploying the
full GitHub Action. Run locally where you have internet access:

    pip install yfinance pandas numpy pytz --break-system-packages
    python test_cupid.py

It prints CMP, the last 4 weekly Heikin-Ashi trend labels, and the latest
weekly StochRSI(14) — compare these against your Chartink chart for CUPID.
"""

import sys
sys.path.insert(0, ".")
from fetch_market_data import (
    to_yf_symbol, fetch_current_price, fetch_weekly_analysis,
)

SYMBOL = "CUPID"

if __name__ == "__main__":
    yf_symbol = to_yf_symbol(SYMBOL)
    price = fetch_current_price(yf_symbol)
    weeks, srsi = fetch_weekly_analysis(yf_symbol)

    print(f"\n{SYMBOL} ({yf_symbol})")
    print(f"CMP: {price}")
    print(f"Weekly trend (week1 -> week4, oldest -> latest): {weeks}")
    print(f"StochRSI(14): {srsi}")
    print("\nCompare 'week4' + StochRSI above against chartink.com/stocks/cupid.html")
    print("(weekly, Heikin-Ashi, 3 years, StochRSI 14 panel).")
