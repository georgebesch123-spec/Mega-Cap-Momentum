"""
Mega-cap momentum strategy — backtest.

Holds the 10 largest S&P 500 companies by market capitalization, reviewed each
January, with systematic sell rules. All metrics are computed from the realized
equity curve, not from summary estimates.

Self-contained: rankings are embedded below, so there is no CSV to upload.
Runs every window in PERIODS from a single price download.

Colab
-----
    !pip install yfinance
    !python backtest.py

Rankings source: finhacker.cz, "Largest 20 S&P 500 Companies by Market Cap".
That source labels each list by YEAR-END. This strategy forms in January, so
every list is shifted forward one year here: the row labelled 2000 is the
source's year-end 1999 ranking — what you would have seen in January 2000.
The `year` column is therefore the FORMATION year.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Every window is run against the same downloaded prices. Reporting more than
# one is deliberate: the verdict depends heavily on the start date, and showing
# that is more honest than picking the flattering window.
PERIODS = [
    (1995, 2025),   # includes the late-90s run-up
    (2000, 2025),   # starts at the dot-com peak
]

INITIAL_CAPITAL = 10_000.0

PORTFOLIO_SIZE = 10      # positions held
TOP_N_POOL = 20          # ranks available for replacements

DRAWDOWN_LIMIT = 0.25    # rule 2: sell if >=25% below trailing 52-week high
STAGNATION_YEARS = 10    # rule 4: minimum holding period before the test
STAGNATION_CAGR = 0.08   # rule 4: sell if position CAGR is below this
CONCENTRATION_CAP = 0.25 # trim back to equal weight above this portfolio share

# ^SP500TR is the S&P 500 TOTAL RETURN index. Holdings use dividend-adjusted
# prices, so they earn total return; benchmarking against ^GSPC (price only)
# would omit ~1.8-2%/yr of dividends and flatter the strategy.
BENCHMARK = "^SP500TR"

RISK_FREE_RATE = 0.025   # approximate average 3-month T-bill. Quote this
                         # alongside any Sharpe figure you report.

# Rule 3 (revenue decline) needs annual fundamentals back to 1995. yfinance
# exposes roughly four years of income statements, which is not enough. Results
# therefore reflect a THREE-rule strategy. Say so in the write-up.
REVENUE_RULE_ENABLED = False


# ----------------------------------------------------------------------
# Rankings — formation year, rank, ticker
# ----------------------------------------------------------------------

RANKINGS_CSV = """\
year,rank,ticker
1990,1,XOM
1990,2,IBM
1990,3,GE
1990,4,BMY
1990,5,MRK
1990,6,KO
1990,7,WMT
1990,8,PG
1990,9,VZ
1990,10,JNJ
1990,11,LLY
1990,12,PEP
1990,13,DIS
1990,14,T
1990,15,MMM
1990,16,AIG
1990,17,BA
1990,18,MCD
1990,19,PFE
1990,20,SLB
1991,1,XOM
1991,2,IBM
1991,3,WMT
1991,4,BMY
1991,5,MRK
1991,6,KO
1991,7,GE
1991,8,PG
1991,9,JNJ
1991,10,VZ
1991,11,PEP
1991,12,LLY
1991,13,BA
1991,14,MMM
1991,15,DIS
1991,16,AIG
1991,17,PFE
1991,18,T
1991,19,SLB
1991,20,MO
1992,1,XOM
1992,2,WMT
1992,3,MRK
1992,4,KO
1992,5,IBM
1992,6,BMY
1992,7,GE
1992,8,JNJ
1992,9,PG
1992,10,PEP
1992,11,PFE
1992,12,LLY
1992,13,VZ
1992,14,FNMA
1992,15,MO
1992,16,AIG
1992,17,MMM
1992,18,BA
1992,19,DIS
1992,20,K
1993,1,XOM
1993,2,WMT
1993,3,KO
1993,4,MRK
1993,5,GE
1993,6,PG
1993,7,BMY
1993,8,PEP
1993,9,JNJ
1993,10,IBM
1993,11,DIS
1993,12,PFE
1993,13,HD
1993,14,VZ
1993,15,FNMA
1993,16,AIG
1993,17,INTC
1993,18,LLY
1993,19,MMM
1993,20,MCD
1994,1,XOM
1994,2,KO
1994,3,WMT
1994,4,GE
1994,5,MRK
1994,6,PG
1994,7,PEP
1994,8,IBM
1994,9,JNJ
1994,10,BMY
1994,11,CVX
1994,12,INTC
1994,13,VZ
1994,14,AIG
1994,15,DIS
1994,16,FNMA
1994,17,PFE
1994,18,MCD
1994,19,MMM
1994,20,T
1995,1,XOM
1995,2,KO
1995,3,GE
1995,4,WMT
1995,5,MRK
1995,6,PG
1995,7,IBM
1995,8,MSFT
1995,9,JNJ
1995,10,CVX
1995,11,PEP
1995,12,BMY
1995,13,INTC
1995,14,AIG
1995,15,DIS
1995,16,PFE
1995,17,VZ
1995,18,HD
1995,19,MCD
1995,20,FNMA
1996,1,XOM
1996,2,KO
1996,3,GE
1996,4,MRK
1996,5,PG
1996,6,JNJ
1996,7,MSFT
1996,8,WMT
1996,9,IBM
1996,10,INTC
1996,11,PEP
1996,12,BMY
1996,13,PFE
1996,14,AIG
1996,15,CVX
1996,16,FNMA
1996,17,MCD
1996,18,LLY
1996,19,DIS
1996,20,VZ
1997,1,KO
1997,2,XOM
1997,3,INTC
1997,4,MSFT
1997,5,GE
1997,6,MRK
1997,7,IBM
1997,8,PG
1997,9,JNJ
1997,10,WMT
1997,11,BMY
1997,12,PFE
1997,13,DIS
1997,14,PEP
1997,15,CVX
1997,16,AIG
1997,17,CSCO
1997,18,LLY
1997,19,FNMA
1997,20,JPM
1998,1,KO
1998,2,MSFT
1998,3,XOM
1998,4,GE
1998,5,INTC
1998,6,MRK
1998,7,PG
1998,8,IBM
1998,9,PFE
1998,10,BMY
1998,11,WMT
1998,12,JNJ
1998,13,LLY
1998,14,VZ
1998,15,DIS
1998,16,AIG
1998,17,C
1998,18,FNMA
1998,19,CSCO
1998,20,PEP
1999,1,MSFT
1999,2,GE
1999,3,INTC
1999,4,WMT
1999,5,XOM
1999,6,KO
1999,7,IBM
1999,8,MRK
1999,9,PFE
1999,10,CSCO
1999,11,BMY
1999,12,PG
1999,13,C
1999,14,JNJ
1999,15,BAC
1999,16,LLY
1999,17,VZ
1999,18,HD
1999,19,AIG
1999,20,T
2000,1,MSFT
2000,2,CSCO
2000,3,WMT
2000,4,GE
2000,5,XOM
2000,6,INTC
2000,7,C
2000,8,IBM
2000,9,HD
2000,10,ORCL
2000,11,KO
2000,12,PG
2000,13,MRK
2000,14,AIG
2000,15,JNJ
2000,16,QCOM
2000,17,BMY
2000,18,PFE
2000,19,T
2000,20,VZ
2001,1,XOM
2001,2,GE
2001,3,CSCO
2001,4,PFE
2001,5,WMT
2001,6,MSFT
2001,7,C
2001,8,INTC
2001,9,MRK
2001,10,AIG
2001,11,ORCL
2001,12,KO
2001,13,JNJ
2001,14,IBM
2001,15,BMY
2001,16,VZ
2001,17,T
2001,18,HD
2001,19,LLY
2001,20,PG
2002,1,MSFT
2002,2,XOM
2002,3,C
2002,4,WMT
2002,5,GE
2002,6,PFE
2002,7,INTC
2002,8,IBM
2002,9,JNJ
2002,10,AIG
2002,11,CSCO
2002,12,VZ
2002,13,MRK
2002,14,HD
2002,15,KO
2002,16,PG
2002,17,BMY
2002,18,T
2002,19,BAC
2002,20,CVX
2003,1,MSFT
2003,2,XOM
2003,3,WMT
2003,4,C
2003,5,PFE
2003,6,JNJ
2003,7,GE
2003,8,IBM
2003,9,AIG
2003,10,MRK
2003,11,PG
2003,12,KO
2003,13,VZ
2003,14,BAC
2003,15,INTC
2003,16,CSCO
2003,17,WFC
2003,18,PEP
2003,19,LLY
2003,20,CVX
2004,1,MSFT
2004,2,XOM
2004,3,PFE
2004,4,C
2004,5,WMT
2004,6,INTC
2004,7,GE
2004,8,CSCO
2004,9,JNJ
2004,10,IBM
2004,11,AIG
2004,12,PG
2004,13,KO
2004,14,BAC
2004,15,WFC
2004,16,MRK
2004,17,VZ
2004,18,CVX
2004,19,UPS
2004,20,HD
2005,1,XOM
2005,2,MSFT
2005,3,C
2005,4,GE
2005,5,WMT
2005,6,BAC
2005,7,PFE
2005,8,JNJ
2005,9,IBM
2005,10,INTC
2005,11,AIG
2005,12,PG
2005,13,JPM
2005,14,CSCO
2005,15,VZ
2005,16,CVX
2005,17,WFC
2005,18,KO
2005,19,UPS
2005,20,HD
2006,1,XOM
2006,2,MSFT
2006,3,C
2006,4,GE
2006,5,WMT
2006,6,BAC
2006,7,JNJ
2006,8,PFE
2006,9,INTC
2006,10,AIG
2006,11,JPM
2006,12,PG
2006,13,CVX
2006,14,IBM
2006,15,CSCO
2006,16,WFC
2006,17,PEP
2006,18,AMGN
2006,19,KO
2006,20,HD
2007,1,XOM
2007,2,MSFT
2007,3,C
2007,4,BAC
2007,5,GE
2007,6,PG
2007,7,WMT
2007,8,JNJ
2007,9,PFE
2007,10,JPM
2007,11,CSCO
2007,12,CVX
2007,13,AIG
2007,14,GOOG
2007,15,IBM
2007,16,WFC
2007,17,INTC
2007,18,KO
2007,19,VZ
2007,20,T
2008,1,XOM
2008,2,MSFT
2008,3,PG
2008,4,GE
2008,5,GOOG
2008,6,CVX
2008,7,WMT
2008,8,T
2008,9,JNJ
2008,10,BAC
2008,11,AAPL
2008,12,CSCO
2008,13,C
2008,14,INTC
2008,15,JPM
2008,16,PFE
2008,17,IBM
2008,18,KO
2008,19,VZ
2008,20,AIG
2009,1,XOM
2009,2,WMT
2009,3,PG
2009,4,MSFT
2009,5,JNJ
2009,6,CVX
2009,7,T
2009,8,JPM
2009,9,PFE
2009,10,IBM
2009,11,KO
2009,12,VZ
2009,13,WFC
2009,14,GOOG
2009,15,GE
2009,16,CSCO
2009,17,ORCL
2009,18,PM
2009,19,PEP
2009,20,INTC
2010,1,XOM
2010,2,MSFT
2010,3,WMT
2010,4,GOOG
2010,5,AAPL
2010,6,JNJ
2010,7,PG
2010,8,JPM
2010,9,IBM
2010,10,CVX
2010,11,WFC
2010,12,PFE
2010,13,CSCO
2010,14,KO
2010,15,BAC
2010,16,T
2010,17,ORCL
2010,18,INTC
2010,19,MRK
2010,20,VZ
2011,1,XOM
2011,2,AAPL
2011,3,MSFT
2011,4,BRK-B
2011,5,WMT
2011,6,GOOG
2011,7,CVX
2011,8,PG
2011,9,IBM
2011,10,JNJ
2011,11,JPM
2011,12,WFC
2011,13,ORCL
2011,14,KO
2011,15,C
2011,16,BAC
2011,17,PFE
2011,18,T
2011,19,GE
2011,20,INTC
2012,1,XOM
2012,2,AAPL
2012,3,MSFT
2012,4,CVX
2012,5,GOOG
2012,6,IBM
2012,7,WMT
2012,8,BRK-B
2012,9,PG
2012,10,JNJ
2012,11,KO
2012,12,PFE
2012,13,WFC
2012,14,PM
2012,15,T
2012,16,ORCL
2012,17,JPM
2012,18,INTC
2012,19,VZ
2012,20,GE
2013,1,AAPL
2013,2,XOM
2013,3,GOOG
2013,4,WMT
2013,5,MSFT
2013,6,BRK-B
2013,7,CVX
2013,8,IBM
2013,9,JNJ
2013,10,PG
2013,11,WFC
2013,12,PFE
2013,13,JPM
2013,14,KO
2013,15,ORCL
2013,16,T
2013,17,PM
2013,18,GE
2013,19,BAC
2013,20,VZ
2014,1,AAPL
2014,2,XOM
2014,3,GOOG
2014,4,MSFT
2014,5,BRK-B
2014,6,JNJ
2014,7,WMT
2014,8,CVX
2014,9,WFC
2014,10,PG
2014,11,JPM
2014,12,IBM
2014,13,PFE
2014,14,AMZN
2014,15,KO
2014,16,ORCL
2014,17,GE
2014,18,BAC
2014,19,C
2014,20,VZ
2015,1,AAPL
2015,2,XOM
2015,3,MSFT
2015,4,BRK-B
2015,5,GOOG
2015,6,JNJ
2015,7,WFC
2015,8,WMT
2015,9,PG
2015,10,JPM
2015,11,META
2015,12,CVX
2015,13,ORCL
2015,14,VZ
2015,15,BAC
2015,16,PFE
2015,17,KO
2015,18,INTC
2015,19,C
2015,20,DIS
2016,1,AAPL
2016,2,GOOG
2016,3,MSFT
2016,4,BRK-B
2016,5,XOM
2016,6,AMZN
2016,7,META
2016,8,JNJ
2016,9,WFC
2016,10,JPM
2016,11,PG
2016,12,WMT
2016,13,PFE
2016,14,VZ
2016,15,KO
2016,16,V
2016,17,GE
2016,18,BAC
2016,19,DIS
2016,20,CVX
2017,1,AAPL
2017,2,GOOG
2017,3,MSFT
2017,4,BRK-B
2017,5,XOM
2017,6,AMZN
2017,7,META
2017,8,JNJ
2017,9,JPM
2017,10,WFC
2017,11,PG
2017,12,BAC
2017,13,CVX
2017,14,VZ
2017,15,WMT
2017,16,T
2017,17,PFE
2017,18,V
2017,19,KO
2017,20,INTC
2018,1,AAPL
2018,2,GOOG
2018,3,MSFT
2018,4,AMZN
2018,5,META
2018,6,BRK-B
2018,7,JNJ
2018,8,JPM
2018,9,XOM
2018,10,BAC
2018,11,WFC
2018,12,WMT
2018,13,V
2018,14,CVX
2018,15,PG
2018,16,HD
2018,17,INTC
2018,18,VZ
2018,19,UNH
2018,20,PFE
2019,1,MSFT
2019,2,AAPL
2019,3,AMZN
2019,4,GOOG
2019,5,BRK-B
2019,6,META
2019,7,JNJ
2019,8,JPM
2019,9,V
2019,10,XOM
2019,11,WMT
2019,12,UNH
2019,13,BAC
2019,14,PFE
2019,15,VZ
2019,16,PG
2019,17,INTC
2019,18,WFC
2019,19,CVX
2019,20,KO
2020,1,AAPL
2020,2,MSFT
2020,3,GOOG
2020,4,AMZN
2020,5,META
2020,6,BRK-B
2020,7,JPM
2020,8,V
2020,9,JNJ
2020,10,WMT
2020,11,BAC
2020,12,PG
2020,13,MA
2020,14,XOM
2020,15,UNH
2020,16,DIS
2020,17,INTC
2020,18,VZ
2020,19,HD
2020,20,KO
2021,1,AAPL
2021,2,MSFT
2021,3,AMZN
2021,4,GOOG
2021,5,META
2021,6,TSLA
2021,7,BRK-B
2021,8,V
2021,9,JNJ
2021,10,WMT
2021,11,JPM
2021,12,MA
2021,13,PG
2021,14,UNH
2021,15,NVDA
2021,16,DIS
2021,17,HD
2021,18,PYPL
2021,19,BAC
2021,20,VZ
2022,1,AAPL
2022,2,MSFT
2022,3,GOOG
2022,4,AMZN
2022,5,TSLA
2022,6,META
2022,7,NVDA
2022,8,BRK-B
2022,9,UNH
2022,10,JPM
2022,11,V
2022,12,JNJ
2022,13,HD
2022,14,WMT
2022,15,PG
2022,16,BAC
2022,17,MA
2022,18,PFE
2022,19,DIS
2022,20,AVGO
2023,1,AAPL
2023,2,MSFT
2023,3,GOOG
2023,4,AMZN
2023,5,BRK-B
2023,6,UNH
2023,7,JNJ
2023,8,XOM
2023,9,V
2023,10,JPM
2023,11,TSLA
2023,12,WMT
2023,13,NVDA
2023,14,PG
2023,15,LLY
2023,16,CVX
2023,17,MA
2023,18,HD
2023,19,META
2023,20,PFE
2024,1,AAPL
2024,2,MSFT
2024,3,GOOG
2024,4,AMZN
2024,5,NVDA
2024,6,META
2024,7,TSLA
2024,8,BRK-B
2024,9,LLY
2024,10,V
2024,11,AVGO
2024,12,JPM
2024,13,UNH
2024,14,WMT
2024,15,MA
2024,16,XOM
2024,17,JNJ
2024,18,PG
2024,19,HD
2024,20,COST
2025,1,AAPL
2025,2,NVDA
2025,3,MSFT
2025,4,GOOG
2025,5,AMZN
2025,6,META
2025,7,TSLA
2025,8,AVGO
2025,9,BRK-B
2025,10,WMT
2025,11,LLY
2025,12,JPM
2025,13,V
2025,14,MA
2025,15,XOM
2025,16,ORCL
2025,17,UNH
2025,18,COST
2025,19,PG
2025,20,HD
2026,1,NVDA
2026,2,AAPL
2026,3,GOOG
2026,4,MSFT
2026,5,AMZN
2026,6,META
2026,7,AVGO
2026,8,TSLA
2026,9,BRK-B
2026,10,LLY
2026,11,WMT
2026,12,JPM
2026,13,V
2026,14,ORCL
2026,15,MA
2026,16,XOM
2026,17,JNJ
2026,18,PLTR
2026,19,ABBV
2026,20,BAC
"""


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_rankings() -> dict[int, list[str]]:
    """Return {formation year: [tickers in rank order]}."""
    df = pd.read_csv(io.StringIO(RANKINGS_CSV)).sort_values(["year", "rank"])
    return {
        int(year): list(group["ticker"].str.strip().str.upper())
        for year, group in df.groupby("year")
    }


def download_prices(tickers: set[str], first_year: int, last_year: int) -> pd.DataFrame:
    """Daily adjusted closes for every ticker that ever appears."""
    symbols = sorted(tickers | {BENCHMARK})
    print(f"Downloading {len(symbols)} symbols, {first_year - 1}–{last_year + 1}...")

    raw = yf.download(
        symbols,
        start=f"{first_year - 1}-01-01",
        end=f"{last_year + 1}-01-10",
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        level = 0 if "Close" in raw.columns.get_level_values(0) else 1
        prices = raw.xs("Close", axis=1, level=level)
    else:
        prices = raw[["Close"]].rename(columns={"Close": symbols[0]})

    empty = [c for c in prices.columns if prices[c].notna().sum() == 0]
    if empty:
        print(f"  WARNING — no data returned for: {empty}")
        print("  Delisted names need a manual series; dropping them biases results upward.")

    short = [c for c in prices.columns
             if prices[c].notna().sum() and prices[c].dropna().index[0].year > first_year]
    if short:
        print(f"  NOTE — price history starts after {first_year} for: {short}")
        print("  These are skipped at formation in years before their data begins.")

    return prices.ffill()


def price_on(prices: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float | None:
    """Last available close at or before `date`."""
    if ticker not in prices.columns:
        return None
    series = prices[ticker].loc[:date].dropna()
    return float(series.iloc[-1]) if len(series) else None


# ----------------------------------------------------------------------
# Sell rules
# ----------------------------------------------------------------------

def hit_drawdown(prices, ticker, review_date) -> bool:
    """Rule 2 — at or below DRAWDOWN_LIMIT off the trailing 52-week high."""
    window = prices[ticker].loc[review_date - pd.Timedelta(days=365):review_date].dropna()
    if len(window) < 2:
        return False
    return window.iloc[-1] <= window.max() * (1 - DRAWDOWN_LIMIT)


def hit_stagnation(position, review_date) -> bool:
    """Rule 4 — held long enough, compounding too slowly."""
    years = (review_date - position["entry_date"]).days / 365.25
    if years < STAGNATION_YEARS:
        return False
    growth = position["price"] / position["entry_price"]
    return growth ** (1 / years) - 1 < STAGNATION_CAGR


def sell_reason(position, ticker, top_n, prices, review_date) -> str | None:
    if ticker not in top_n[:PORTFOLIO_SIZE]:
        return "rank_exit"
    if hit_drawdown(prices, ticker, review_date):
        return "drawdown"
    if REVENUE_RULE_ENABLED:
        raise NotImplementedError("Wire in a fundamentals source before enabling rule 3.")
    if hit_stagnation(position, review_date):
        return "stagnation"
    return None


def pick_replacement(top_n, held, prices, review_date, excluded=frozenset()) -> str | None:
    """Largest company not held, subject to a positive prior-year return.

    `excluded` holds names sold earlier in this same review, so a position
    dropped on a rule cannot be repurchased moments later.
    """
    prior = review_date - pd.Timedelta(days=365)
    for ticker in top_n:
        if ticker in held or ticker in excluded or ticker not in prices.columns:
            continue
        then, now = price_on(prices, ticker, prior), price_on(prices, ticker, review_date)
        if then and now and now > then:
            return ticker
    return None


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------

def run(rankings, prices, start_year, end_year):
    positions: dict[str, dict] = {}
    cash = 0.0            # proceeds with no eligible replacement wait here
    equity, trades = {}, []

    for year in range(start_year, end_year + 1):
        review = prices.index[prices.index >= f"{year}-01-01"][0]
        top_n = rankings[year][:TOP_N_POOL]
        sold_this_review: set[str] = set()

        for ticker, pos in positions.items():
            px = price_on(prices, ticker, review)
            if px:
                pos["price"] = px

        if not positions:
            per_name = INITIAL_CAPITAL / PORTFOLIO_SIZE
            for ticker in top_n[:PORTFOLIO_SIZE]:
                px = price_on(prices, ticker, review)
                if not px:
                    print(f"  {year}: no price for {ticker}, skipped at formation")
                    continue
                positions[ticker] = {"shares": per_name / px, "price": px,
                                     "entry_price": px, "entry_date": review}
        else:
            for ticker in list(positions):
                reason = sell_reason(positions[ticker], ticker, top_n, prices, review)
                if not reason:
                    continue

                proceeds = positions[ticker]["shares"] * positions[ticker]["price"]
                del positions[ticker]
                sold_this_review.add(ticker)

                new = pick_replacement(top_n, set(positions), prices, review,
                                       excluded=sold_this_review)
                trades.append({"year": year, "sold": ticker, "reason": reason,
                               "bought": new or "(to cash — no eligible name)"})
                if not new:
                    cash += proceeds          # held, not destroyed
                    continue

                px = price_on(prices, new, review)
                positions[new] = {"shares": proceeds / px, "price": px,
                                  "entry_price": px, "entry_date": review}

            # redeploy any cash parked from an earlier review
            while cash > 0:
                new = pick_replacement(top_n, set(positions), prices, review,
                                       excluded=sold_this_review)
                if not new:
                    break
                px = price_on(prices, new, review)
                positions[new] = {"shares": cash / px, "price": px,
                                  "entry_price": px, "entry_date": review}
                trades.append({"year": year, "sold": "(cash)",
                               "reason": "redeploy", "bought": new})
                cash = 0.0

            # trim only a position that has grown past the concentration cap
            total = sum(p["shares"] * p["price"] for p in positions.values())
            for ticker, pos in positions.items():
                value = pos["shares"] * pos["price"]
                if value / total > CONCENTRATION_CAP and len(positions) > 1:
                    target = total / len(positions)
                    excess = value - target
                    pos["shares"] = target / pos["price"]
                    per_other = excess / (len(positions) - 1)
                    for other, op in positions.items():
                        if other != ticker:
                            op["shares"] += per_other / op["price"]
                    trades.append({"year": year, "sold": f"{ticker} (trim)",
                                   "reason": "concentration", "bought": "pro-rata"})
                    break

        year_end = prices.index[prices.index <= f"{year}-12-31"][-1]
        equity[year] = cash + sum(
            pos["shares"] * (price_on(prices, t, year_end) or pos["price"])
            for t, pos in positions.items()
        )

    return pd.Series(equity, name="equity"), pd.DataFrame(trades)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def metrics(equity: pd.Series, initial: float = INITIAL_CAPITAL) -> dict:
    """Everything computed from the realized curve. No estimates."""
    values = pd.concat([pd.Series({equity.index[0] - 1: initial}), equity])
    returns = values.pct_change().dropna()
    years = len(returns)

    cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    arithmetic = returns.mean()
    vol = returns.std(ddof=1)

    downside = returns[returns < RISK_FREE_RATE]
    downside_dev = np.sqrt(((downside - RISK_FREE_RATE) ** 2).sum() / years)

    running_max = values.cummax()
    max_dd = ((values - running_max) / running_max).min()

    return {
        "CAGR": cagr,
        "Arithmetic mean": arithmetic,
        "Volatility": vol,
        "Sharpe": (arithmetic - RISK_FREE_RATE) / vol,
        "Sortino": (arithmetic - RISK_FREE_RATE) / downside_dev if downside_dev else np.nan,
        "Max drawdown (annual)": max_dd,
        "Calmar": cagr / abs(max_dd) if max_dd else np.nan,
        "Positive years": (returns > 0).mean(),
        "Final value": values.iloc[-1],
    }


def benchmark_equity(prices: pd.DataFrame, start_year: int, end_year: int) -> pd.Series:
    out, shares = {}, None
    for year in range(start_year, end_year + 1):
        if shares is None:
            start = prices.index[prices.index >= f"{year}-01-01"][0]
            shares = INITIAL_CAPITAL / price_on(prices, BENCHMARK, start)
        year_end = prices.index[prices.index <= f"{year}-12-31"][-1]
        out[year] = shares * price_on(prices, BENCHMARK, year_end)
    return pd.Series(out)


def comparison_table(strategy: dict, bench: dict) -> str:
    rows, pct = [], {"CAGR", "Arithmetic mean", "Volatility",
                     "Max drawdown (annual)", "Positive years"}
    rows.append(f"  {'Metric':<24}{'Strategy':>14}{'S&P 500 TR':>14}")
    rows.append(f"  {'-'*24}{'-'*14}{'-'*14}")
    for k in strategy:
        s, b = strategy[k], bench[k]
        if k == "Final value":
            rows.append(f"  {k:<24}{'$'+format(s, ',.0f'):>14}{'$'+format(b, ',.0f'):>14}")
        elif k in pct:
            rows.append(f"  {k:<24}{s:>13.2%}{b:>14.2%}")
        else:
            rows.append(f"  {k:<24}{s:>14.2f}{b:>14.2f}")
    return "\n".join(rows)


# ----------------------------------------------------------------------

def main() -> None:
    rankings = load_rankings()

    first = min(p[0] for p in PERIODS)
    last = max(p[1] for p in PERIODS)
    universe = {t for y, names in rankings.items()
                if first <= y <= last
                for t in names[:TOP_N_POOL]}
    prices = download_prices(universe, first, last)

    for start_year, end_year in PERIODS:
        missing = [y for y in range(start_year, end_year + 1) if y not in rankings]
        if missing:
            print(f"\nSkipping {start_year}–{end_year}: no rankings for {missing}")
            continue

        equity, trades = run(rankings, prices, start_year, end_year)
        strat = metrics(equity)
        bench = metrics(benchmark_equity(prices, start_year, end_year))

        header = f"{start_year}–{end_year}  ({end_year - start_year + 1} years)"
        print(f"\n{'=' * 54}\n{header}\n{'=' * 54}")
        print(comparison_table(strat, bench))

        gap = strat["CAGR"] - bench["CAGR"]
        verdict = "outperformed" if gap > 0 else "underperformed"
        print(f"\n  Strategy {verdict} by {abs(gap):.2%} a year on compound return.")
        print(f"  Sharpe {strat['Sharpe']:.2f} vs {bench['Sharpe']:.2f}"
              f"  (risk-free {RISK_FREE_RATE:.1%})")

        print(f"\n  Rule firings: {len(trades)}")
        if not trades.empty:
            for reason, n in trades.groupby("reason").size().items():
                print(f"    {reason:<16}{n:>4}")

        tag = f"{start_year}_{end_year}"
        equity.to_csv(f"equity_{tag}.csv", header=["portfolio_value"])
        trades.to_csv(f"trades_{tag}.csv", index=False)

    print("\nWrote equity_*.csv and trades_*.csv for each period.")
    print("Rule 3 (revenue decline) is OFF: this is a three-rule strategy.")
    print("Benchmark is ^SP500TR (total return), matching dividend-adjusted holdings.")


if __name__ == "__main__":
    main()
