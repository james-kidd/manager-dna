"""
Stage 2: GMM Regime Modeling

Fits a Gaussian Mixture Model over macro features to classify market regimes.
Based on Two Sigma's "A Machine Learning Approach to Regime Modeling."

Mathematical basis:
    p(X_t) = Σ_{k=1}^{K} π_k · N(X_t | μ_k, Σ_k)
"""

import datetime

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

MACRO_FEATURES = ["SPY_Return", "VIX_Level", "Credit_Spread", "10Y_Yield_Change"]


class MarketRegimeModel:
    def __init__(self, start_date="2018-01-01", end_date=None, n_regimes=3):
        self.start_date = start_date
        self.end_date = end_date or datetime.datetime.today().strftime("%Y-%m-%d")
        self.n_regimes = n_regimes
        self.gmm = GaussianMixture(n_components=n_regimes, covariance_type="full", random_state=42)
        self.scaler = StandardScaler()
        self.data = pd.DataFrame()

    def fetch_macro_data(self):
        print("Fetching market and macro data...")

        tickers = ["SPY", "^VIX"]
        raw = yf.download(tickers, start=self.start_date, end=self.end_date)
        df_yf = raw["Adj Close"] if "Adj Close" in raw.columns.get_level_values(0) else raw["Close"]
        df_yf["SPY_Return"] = np.log(df_yf["SPY"] / df_yf["SPY"].shift(1))
        df_yf["VIX_Level"] = df_yf["^VIX"]

        df_fred = web.DataReader(["BAMLH0A0HYM2", "DGS10"], "fred", self.start_date, self.end_date)
        df_fred.columns = ["Credit_Spread", "10Y_Yield"]
        df_fred["10Y_Yield_Change"] = df_fred["10Y_Yield"].diff()

        self.data = df_yf[["SPY_Return", "VIX_Level", "SPY"]].join(
            df_fred[["Credit_Spread", "10Y_Yield_Change"]], how="inner"
        )
        self.data.dropna(inplace=True)
        print(f"Macro data merged: {len(self.data)} trading days.")

    def fit_predict_regimes(self):
        X = self.data[MACRO_FEATURES]
        X_scaled = self.scaler.fit_transform(X)

        print(f"Fitting GMM with {self.n_regimes} hidden regimes...")
        self.gmm.fit(X_scaled)

        self.data["Regime"] = self.gmm.predict(X_scaled)
        probs = self.gmm.predict_proba(X_scaled)
        self.data["Regime_Probability"] = probs.max(axis=1)

        return self.data

    def get_regime_summary(self):
        means = self.data.groupby("Regime")[MACRO_FEATURES].mean()
        counts = self.data["Regime"].value_counts().sort_index().rename("N_Days")
        share = (counts / counts.sum()).rename("Share")
        return means.join(counts).join(share)

    def plot_regimes(self, save_path="output/market_regimes_gmm.png"):
        fig, ax = plt.subplots(figsize=(14, 7))
        self.data["Cum_Return"] = (1 + self.data["SPY_Return"]).cumprod()

        colors = ["green", "red", "orange", "blue", "purple"]
        for regime in range(self.n_regimes):
            regime_data = self.data[self.data["Regime"] == regime]
            ax.scatter(regime_data.index, regime_data["Cum_Return"],
                       color=colors[regime], label=f"Regime {regime}", s=10)

        ax.plot(self.data.index, self.data["Cum_Return"], color="black", alpha=0.3, linewidth=1)
        plt.title("S&P 500 Cumulative Returns by GMM Market Regime")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Regime plot saved: {save_path}")
        plt.close()
