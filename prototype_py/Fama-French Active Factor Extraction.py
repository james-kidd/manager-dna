import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import warnings
warnings.filterwarnings("ignore")

class ManagerialFactorExtractor:
    def __init__(self, fund_ticker, benchmark_ticker='SPY', start_date='2018-01-01', window=63):
        """
        Initialize the Factor Extractor.
        window=63 corresponds to a rolling 3-month (approx. 63 trading days) regression 
        to capture dynamic shifts in manager behavior.
        """
        self.fund_ticker = fund_ticker
        self.benchmark_ticker = benchmark_ticker
        self.start_date = start_date
        self.window = window
        self.data = pd.DataFrame()
        self.factor_loadings = pd.DataFrame()
        
    def fetch_data(self):
        """
        Fetch Fund returns, Benchmark returns, and Fama-French 5-Factor Data.
        """
        print(f"Fetching data for {self.fund_ticker} vs {self.benchmark_ticker}...")
        
        # 1. Fetch Fund and Benchmark Data
        tickers = [self.fund_ticker, self.benchmark_ticker]
        prices = yf.download(tickers, start=self.start_date)['Adj Close']
        
        # Calculate daily returns
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Calculate Active Return (The dependent variable: Y)
        returns['Active_Return'] = returns[self.fund_ticker] - returns[self.benchmark_ticker]
        
        # 2. Fetch Fama-French 5-Factor Daily Data
        # 'F-F_Research_Data_5_Factors_2x3_daily'
        print("Fetching Fama-French 5-Factor Daily Data...")
        ff_dict = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start=self.start_date)
        ff_data = ff_dict[0] / 100.0  # Convert percentages to decimals
        
        # Ensure indices match (timezone naive, matching dates)
        returns.index = returns.index.tz_localize(None)
        ff_data.index = pd.to_datetime(ff_data.index.astype(str))
        
        # Merge datasets
        self.data = returns[['Active_Return']].join(ff_data, how='inner').dropna()
        print("Data successfully merged.")
        
    def extract_rolling_factors(self):
        """
        Run a rolling OLS regression to extract the active factor loadings over time.
        Model: Active_Return = alpha + B1*Mkt-RF + B2*SMB + B3*HML + B4*RMW + B5*CMA
        """
        print(f"Running rolling Fama-French regression (Window: {self.window} days)...")
        
        # Define endogenous (Y) and exogenous (X) variables
        Y = self.data['Active_Return']
        
        # The Fama-French factors (Mkt-RF is the market premium)
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        X = self.data[factors]
        X = sm.add_constant(X) # Add alpha intercept
        
        # Execute Rolling OLS
        model = RollingOLS(endog=Y, exog=X, window=self.window)
        rolling_res = model.fit()
        
        # Extract the coefficients (betas)
        self.factor_loadings = rolling_res.params.dropna()
        
        # Rename columns for clarity
        self.factor_loadings.rename(columns={'const': 'Active_Alpha'}, inplace=True)
        
        print("Factor extraction complete.")
        return self.factor_loadings

# --- Execution Example ---
if __name__ == "__main__":
    # Example using ARKK (Innovation ETF) as our Active Manager proxy
    # ARKK is a prime candidate for distinct managerial DNA (High Beta, Low Value)
    extractor = ManagerialFactorExtractor(fund_ticker='ARKK', benchmark_ticker='SPY', start_date='2018-01-01')
    extractor.fetch_data()
    arkk_dna = extractor.extract_rolling_factors()
    
    # Display the most recent factor loadings to observe the manager's current "Style"
    print("\nMost Recent Active Factor Loadings (Managerial DNA Signature):")
    print(arkk_dna.tail(1).T)