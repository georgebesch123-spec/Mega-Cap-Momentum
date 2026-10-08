# Mega-Cap Momentum Strategy

A falsification test of a rules-based mega-cap equity strategy, 1995–2025.

**Result: the hypothesis is rejected in both windows tested.** The strategy did
not produce excess risk-adjusted return over a passive index position.

## Hypothesis

Market-capitalization rank is backward-looking, but *changes* in rank may not be.
Companies entering the top ten typically arrive on sustained revenue and earnings
growth; companies exiting have typically begun a structural decline. If either
effect persists past the point of the rank change, a portfolio holding the current
top ten and systematically removing deteriorating constituents should beat the index.

The test is deliberately mechanical — no view on any individual business, no
discretion — so that what is tested is whether the rank signal alone carries
information.

## Results

Benchmark is the S&P 500 **Total Return** index (`^SP500TR`), matching the
dividend-adjusted prices the holdings earn. Risk-free rate 2.5%.

### 1995–2025 (31 years)

| Metric | Strategy | S&P 500 TR |
|---|---:|---:|
| CAGR | 11.03% | 11.14% |
| Final value ($10k start) | $256,287 | $264,464 |
| Arithmetic mean | 13.39% | 12.76% |
| Volatility | 22.41% | 18.06% |
| Sharpe | 0.49 | 0.57 |
| Sortino | 0.98 | 1.05 |
| Max drawdown | −45.55% | −37.61% |
| Calmar | 0.24 | 0.30 |
| Positive years | 74.19% | 80.65% |

### 2000–2025 (26 years)

| Metric | Strategy | S&P 500 TR |
|---|---:|---:|
| CAGR | 5.98% | 8.11% |
| Final value ($10k start) | $45,296 | $76,022 |
| Arithmetic mean | 7.90% | 9.72% |
| Volatility | 19.87% | 17.93% |
| Sharpe | 0.27 | 0.40 |
| Sortino | 0.45 | 0.68 |
| Max drawdown | −44.34% | −37.01% |
| Calmar | 0.13 | 0.22 |
| Positive years | 73.08% | 76.92% |

## Where the return went

The 1995–2025 window is the instructive one. The strategy earned a **higher
arithmetic mean** than the index and still finished behind on compound return.
Compound return falls below the arithmetic mean by roughly half the variance:

| | Strategy | S&P 500 TR |
|---|---:|---:|
| Arithmetic mean | 13.39% | 12.76% |
| Less variance drag (σ²/2) | −2.51% | −1.63% |
| Implied compound return | 10.88% | 11.13% |
| Actual CAGR | 11.03% | 11.14% |

The approximation lands within 0.15 points of the realized figures. The strategy
earned 0.63 points a year more than the index and surrendered 0.88 points more to
its own dispersion. The edge was real; it was not large enough to pay for the
volatility used to generate it.

Two specific failures show up in the trade log:

**The drawdown rule is pro-cyclical.** It fires *after* a decline, so it sells into
weakness, and it cannot tell a cyclical drawdown from a structural one at the moment
it fires. January 2001: sold MSFT, CSCO, GE, INTC ahead of the dot-com collapse —
exactly what it was built for. January 2023: sold MSFT, GOOG, AMZN, AAPL in one
review, immediately before the year those names drove the market. January 2009:
three positions went to cash because nothing passed the positive-prior-year filter
at the bottom of the crisis.

**Rank exit drives two-thirds of turnover** — 54 of 79 firings in the longer window.
A company at rank 10 and one at rank 11 differ negligibly, yet the rule treats the
boundary as decisive. Much of this is noise.

## Strategy

Hold the ten largest S&P 500 companies by market capitalization. Review each January.

| # | Rule | Trigger | Status |
|---|---|---|---|
| 1 | Rank exit | No longer in the top ten | Tested |
| 2 | Drawdown | 25%+ below trailing 52-week high | Tested |
| 3 | Revenue decline | Revenue contracted year over year | **Not tested** — see below |
| 4 | Stagnation | Held 10+ years at position CAGR below 8% | Tested |

**Buy rule** — replace with the largest company not already held, subject to a
positive prior-calendar-year return, excluding anything sold earlier in the same
review. Where nothing qualifies, proceeds go to cash and are redeployed next January.

**Weighting** — equal at formation, then left to drift. Trimmed back to equal weight
only above 25% of portfolio value.

## Running it

```bash
pip install yfinance pandas numpy
python backtest.py
```

Self-contained — rankings for formation years 1990–2026 are embedded, so there is
nothing to download or upload. Both windows run from a single price fetch; edit
`PERIODS` to add others. Takes roughly 15–20 minutes, almost all of it downloading.

Outputs `equity_<period>.csv` and `trades_<period>.csv` per window, with the
triggering rule recorded for every firing.

## Methodology notes

**Point-in-time rankings.** The source (FinHacker) labels each list by year-end;
the strategy forms in January, so every list is shifted forward one year — the row
labelled 2000 is the year-end 1999 ranking, which is what a contemporaneous investor
would have seen. These are not reconstructed from present-day knowledge of which
companies grew large, so the backtest is free of look-ahead bias in constituent
selection.

**Total return, both sides.** Holdings use dividend-adjusted prices. Benchmarking
against `^GSPC` (price only) would omit ~1.8–2%/yr of dividends and flatter the
strategy by roughly that much. `^SP500TR` is the correct comparison.

**Rule 3 is not backtested.** Annual revenue back to 1995 is not available through
yfinance, which exposes roughly four years of income statements. The rule is
implemented as a hook and disabled (`REVENUE_RULE_ENABLED = False`). Everything
reported reflects a **three-rule** strategy, and the fourth rule's contribution is
unknown. A fundamentals-based exit is the most plausible fix for the pro-cyclical
failure above, which makes this the most important untested piece.

**Metrics are computed, not estimated.** Arithmetic mean, standard deviation, Sharpe,
Sortino, maximum drawdown and Calmar all derive from the realized annual equity curve.
The risk-free rate is set explicitly in the config block and should be quoted alongside
any Sharpe figure.

**Drawdown is annual-resolution.** The review is annual, so the equity curve is annual
and intra-year drawdowns are not captured. True peak-to-trough was deeper than reported,
for both the strategy and the index.

**Costs excluded.** No transaction costs, spread or taxes. Turnover is under three
trades a year so trading costs are immaterial; taxes are not. In a taxable account the
realized gains at each January review would widen the gap to a buy-and-hold index
position, not narrow it.

**Price history gaps.** Twelve tickers have histories beginning after 1995, mostly
companies not yet listed. The script reports them rather than failing silently, and
they appear in the rankings only in years after their data begins.

## Limitations

Both windows end in 2025 and overlap heavily, so they are not independent tests.
Thirty-one annual observations with 79 rule firings is a small effective sample and a
handful of events drive most of the outcome. Agreement between the two windows is
reassuring but well short of statistical significance.

The universe is S&P 500 constituents, so a large non-index company would be excluded.

The 25% drawdown threshold was chosen without optimization and has not been varied.

## What to test next

- **Buffer the rank-exit rule** — sell only on a fall past rank 12 or 13 rather than at
  the top-ten boundary. This rule drives two-thirds of turnover, much of it noise.
- **Vary the drawdown threshold** across 15–40% to see whether the result is robust or
  an artifact of one number.
- **Source annual fundamentals and enable rule 3**, then retest.
- **Widen the portfolio to 15 or 20 names.** Variance, not selection, is the binding
  constraint in the longer window; more positions reduce dispersion directly.
- **Extend to 1990** — the embedded rankings already support it — and run a walk-forward
  split, fitting on the earlier period and testing on data not examined while designing
  the rules.

## Files

```
backtest.py              strategy, simulation, metrics, embedded rankings
mega-cap-momentum.pdf    full write-up
equity_<period>.csv      output — annual portfolio values
trades_<period>.csv      output — every rule firing with its reason
```
