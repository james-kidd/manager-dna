"""
Stage 1: Fama-French Active Factor Extraction

Rolling OLS of active return (fund - benchmark) against the FF 5-Factor model.
Produces a time-series of rolling [alpha, Mkt-RF, SMB, HML, RMW, CMA] betas —
the fund's shifting style fingerprint over time.

Mathematical basis:
    ΔR_{f,t} = α_f + β₁(MKT) + β₂(SMB) + β₃(HML) + β₄(RMW) + β₅(CMA) + ε
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import warnings

warnings.filterwarnings("ignore")

FF_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


class ManagerialFactorExtractor:
    def __init__(self, fund_ticker, benchmark_ticker="SPY", start_date="2018-01-01", window=63):
        self.fund_ticker = fund_ticker
        self.benchmark_ticker = benchmark_ticker
        self.start_date = start_date
        self.window = window
        self.data = pd.DataFrame()
        self.factor_loadings = pd.DataFrame()

    def fetch_data(self):
        print(f"Fetching data for {self.fund_ticker} vs {self.benchmark_ticker}...")

        tickers = [self.fund_ticker, self.benchmark_ticker]
        raw = yf.download(tickers, start=self.start_date)
        prices = raw["Adj Close"] if "Adj Close" in raw.columns.get_level_values(0) else raw["Close"]
        returns = np.log(prices / prices.shift(1)).dropna()
        returns["Active_Return"] = returns[self.fund_ticker] - returns[self.benchmark_ticker]

        print("Fetching Fama-French 5-Factor Daily Data...")
        ff_dict = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=self.start_date)
        ff_data = ff_dict[0] / 100.0

        returns.index = returns.index.tz_localize(None)
        ff_data.index = pd.to_datetime(ff_data.index.astype(str))

        self.data = returns[["Active_Return"]].join(ff_data, how="inner").dropna()
        print(f"Data merged: {len(self.data)} trading days.")

    def extract_rolling_factors(self):
        print(f"Running rolling FF regression (window={self.window} days)...")

        Y = self.data["Active_Return"]
        X = sm.add_constant(self.data[FF_FACTORS])

        model = RollingOLS(endog=Y, exog=X, window=self.window)
        rolling_res = model.fit()

        self.factor_loadings = rolling_res.params.dropna()
        self.factor_loadings.rename(columns={"const": "Active_Alpha"}, inplace=True)

        print(f"Factor extraction complete: {len(self.factor_loadings)} observations.")
        return self.factor_loadings

    def get_latest_signature(self):
        if self.factor_loadings.empty:
            raise ValueError("Run extract_rolling_factors() first.")
        return self.factor_loadings.iloc[-1]
