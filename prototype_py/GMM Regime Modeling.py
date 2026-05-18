import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime

class MarketRegimeModel:
    def __init__(self, start_date='2018-01-01', end_date=datetime.datetime.today().strftime('%Y-%m-%d'), n_regimes=3):
        """
        Initialize the GMM Regime Model based on the Two Sigma framework.
        n_regimes=3 typically represents: Bull/Low-Vol, Bear/High-Vol, and Transition.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.n_regimes = n_regimes
        self.gmm = GaussianMixture(n_components=self.n_regimes, covariance_type='full', random_state=42)
        self.scaler = StandardScaler()
        self.data = pd.DataFrame()
        
    def fetch_macro_data(self):
        """
        Fetch multidimensional features to represent the state of the market.
        1. VIX: Implied Volatility (Fear gauge)
        2. SPY Returns: Broad market momentum
        3. High Yield Spread (FRED): Credit market stress
        4. 10Y Treasury Yield (FRED): Interest rate environment
        """
        print("Fetching market and macro data...")
        
        # 1. Equity & Volatility Data (Yahoo Finance)
        tickers = ["SPY", "^VIX"]
        df_yf = yf.download(tickers, start=self.start_date, end=self.end_date)['Adj Close']
        df_yf['SPY_Return'] = np.log(df_yf['SPY'] / df_yf['SPY'].shift(1))
        df_yf['VIX_Level'] = df_yf['^VIX']
        
        # 2. Credit & Rates Data (FRED API via pandas_datareader)
        # BAMLH0A0HYM2: US High Yield Option-Adjusted Spread
        # DGS10: 10-Year Treasury Constant Maturity Rate
        df_fred = web.DataReader(['BAMLH0A0HYM2', 'DGS10'], 'fred', self.start_date, self.end_date)
        df_fred.columns = ['Credit_Spread', '10Y_Yield']
        df_fred['10Y_Yield_Change'] = df_fred['10Y_Yield'].diff()
        
        # Merge and clean
        self.data = df_yf[['SPY_Return', 'VIX_Level', 'SPY']].join(df_fred[['Credit_Spread', '10Y_Yield_Change']], how='inner')
        self.data.dropna(inplace=True)
        print("Data successfully fetched and merged.")

    def fit_predict_regimes(self):
        """
        Scale the features and fit the Gaussian Mixture Model to define the regimes.
        """
        # Select our feature tensor X_t
        features = ['SPY_Return', 'VIX_Level', 'Credit_Spread', '10Y_Yield_Change']
        X = self.data[features]
        
        # Standardization is critical for GMM so variances of different magnitudes don't dominate
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Fitting GMM with {self.n_regimes} hidden regimes...")
        self.gmm.fit(X_scaled)
        
        # Predict the regime (0, 1, or 2) and extract the probabilities
        self.data['Regime'] = self.gmm.predict(X_scaled)
        
        # Get the probability of being in the assigned regime (confidence level)
        probs = self.gmm.predict_proba(X_scaled)
        self.data['Regime_Probability'] = probs.max(axis=1)
        
        return self.data
    
    def plot_regimes(self):
        """
        Visualize the SPY cumulative returns colored by the hidden market regime.
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Calculate cumulative returns for plotting
        self.data['Cum_Return'] = (1 + self.data['SPY_Return']).cumprod()
        
        colors = ['green', 'red', 'orange', 'blue', 'purple']
        
        for regime in range(self.n_regimes):
            # Mask to isolate dates corresponding to a specific regime
            regime_data = self.data[self.data['Regime'] == regime]
            ax.scatter(regime_data.index, regime_data['Cum_Return'], 
                       color=colors[regime], label=f'Regime {regime}', s=10)
            
        ax.plot(self.data.index, self.data['Cum_Return'], color='black', alpha=0.3, linewidth=1)
        
        plt.title('S&P 500 Cumulative Returns Colored by GMM Market Regime')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("market_regimes_gmm.png")
        print("Plot saved as market_regimes_gmm.png")

# --- Execution ---
if __name__ == "__main__":
    regime_model = MarketRegimeModel(start_date='2018-01-01', n_regimes=3)
    regime_model.fetch_macro_data()
    regime_data = regime_model.fit_predict_regimes()
    
    # Analyze the characteristics of each regime
    summary = regime_data.groupby('Regime')[['SPY_Return', 'VIX_Level', 'Credit_Spread']].mean()
    print("\nRegime Characteristics (Averages):")
    print(summary)
    
    regime_model.plot_regimes()