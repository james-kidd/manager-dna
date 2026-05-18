"""
Hull delta-equivalent adjustment for options-holding funds.

For an option position the delta-equivalent share exposure is

    Δ_eq = Δ × Multiplier × Contracts

(Hull, "Options, Futures, and Other Derivatives", Ch. 19). The fund's
effective equity beta to a factor f is then

    β_f^eff = β_f^direct + (Δ_eq · S) / NAV · β_f^underlying

This module provides:

1. Black-Scholes delta via math.erf (no scipy dependency).
2. A FundOptionsExposure spec for declaring per-fund options positions.
3. apply_hull_adjustment(loadings, exposure) which scales rolling FF beta
   loadings by the time-invariant delta-equivalent multiplier.

Data note: ETF options holdings are reported quarterly in SEC Form N-PORT
with a ~60-day lag. There is no free real-time API. This module accepts
user-supplied positions; populating it is a separate data-engineering task.
"""

from dataclasses import dataclass, field
from math import erf, log, sqrt
from typing import Iterable

import pandas as pd


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_delta(spot: float, strike: float, T: float, r: float, sigma: float,
             option_type: str = "call") -> float:
    """Black-Scholes delta. T in years, r and sigma annualized.

    Returns 1.0 / -1.0 / 0.0 at the boundaries (T <= 0 or sigma <= 0).
    """
    if T <= 0 or sigma <= 0:
        if option_type.lower() == "call":
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0

    d1 = (log(spot / strike) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    if option_type.lower() == "call":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


@dataclass
class OptionPosition:
    """A single options position held by a fund.

    delta_equivalent_notional = delta × multiplier × contracts × spot
    Sign convention: long calls and short puts are positive (long equity exposure);
    short calls and long puts are negative.
    """
    underlying: str
    option_type: str  # "call" or "put"
    is_long: bool
    contracts: float
    multiplier: float
    spot: float
    strike: float
    days_to_expiry: float
    iv: float                 # annualized volatility
    risk_free_rate: float = 0.04

    def delta_equivalent_notional(self) -> float:
        T = max(self.days_to_expiry / 365.0, 0.0)
        d = bs_delta(self.spot, self.strike, T, self.risk_free_rate, self.iv, self.option_type)
        sign = 1.0 if self.is_long else -1.0
        return sign * d * self.multiplier * self.contracts * self.spot


@dataclass
class FundOptionsExposure:
    """All option positions held by a single fund, plus the fund's NAV."""
    ticker: str
    nav: float
    positions: list = field(default_factory=list)

    def add(self, position: OptionPosition) -> None:
        self.positions.append(position)

    def adjustment_multiplier(self) -> float:
        """Returns 1 + (total delta-equivalent notional / NAV).

        A value of 1.00 means no adjustment. 1.15 means the fund's true
        equity exposure is 15% above its reported direct holdings. 0.85
        means options are reducing effective equity beta by 15% (typical
        for covered-call funds like JEPI / QYLD).
        """
        if self.nav <= 0:
            raise ValueError(f"NAV must be positive (got {self.nav} for {self.ticker})")
        total = sum(p.delta_equivalent_notional() for p in self.positions)
        return 1.0 + total / self.nav


def apply_hull_adjustment(
    loadings: pd.DataFrame,
    exposure: FundOptionsExposure,
    factor_cols: Iterable[str] = ("Mkt-RF", "SMB", "HML", "RMW", "CMA"),
) -> pd.DataFrame:
    """Scale FF beta loadings by the delta-equivalent multiplier.

    Returns a copy of `loadings` with the factor columns scaled. The
    alpha column (if present) is left untouched -- options don't add alpha,
    they re-shape factor exposure.
    """
    mult = exposure.adjustment_multiplier()
    adj = loadings.copy()
    for col in factor_cols:
        if col in adj.columns:
            adj[col] = adj[col] * mult
    adj.attrs["hull_multiplier"] = mult
    adj.attrs["hull_ticker"] = exposure.ticker
    return adj
