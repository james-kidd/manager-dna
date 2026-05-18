# Research & Mathematical Foundations

## 1. Dimensionality Reduction via Fama-French
Instead of building an $N \times M$ matrix of 3,000 individual stocks, we reduce the decision space to 5 established financial factors.
* **Formula (Active OLS):** $\Delta R_{f,t} = \alpha_f + \beta_1(MKT) + \beta_2(SMB) + \beta_3(HML) + \beta_4(RMW) + \beta_5(CMA) + \epsilon$
* *Insight:* By analyzing active $\beta$s, we separate deliberate managerial "tilts" from general market beta. 

## 2. Regime Modeling via Gaussian Mixture Models (GMM)
*Reference: Two Sigma - "A Machine Learning Approach to Regime Modeling"*
* **Formula:** $p(X_t) = \sum_{k=1}^K \pi_k \mathcal{N}(X_t \mid \mu_k, \Sigma_k)$
* *Insight:* Unlike K-Means, GMM assumes the data is generated from a mixture of finite Gaussian distributions with unknown parameters. It outputs a *probability* of being in a regime, which is mathematically robust for noisy financial features like credit spreads and VIX.

## 3. Managerial Style Extraction via PCA
* *Process:* We apply Principal Component Analysis to the covariance matrix of the funds' active $\beta$ loadings.
* *Insight:* The first 2-3 Principal Components (PCs) represent the orthogonal "Super-Styles" of the market (e.g., PC1 = Extreme Growth/Momentum; PC2 = Deep Value/Quality). A fund's eigenvectors dictate its loading on these components.

## 4. Clustering via Bipartite Graphs
*Reference: Stanford University - "Classification of Assets and Investors in a Financial Bipartite Graph"*
* **Structure:** Graph $G = (U, V, E)$ where $U$ = Funds, $V$ = Super-Styles (PCs), and $E$ = Conviction Weight (PCA loading).
* *Insight:* By mapping this as a bipartite network, we can use edge weights to group visually dissimilar funds into the same behavioral cluster. If the graph topology changes violently between GMM regimes, the manager is highly tactical.

## 5. Derivatives Adjustment (The Hull Constraint)
*Reference: Hull - "Options, Futures, and Other Derivatives"*
* *Constraint:* Funds utilizing covered calls/puts distort raw equity allocations.
* *Adjustment:* Options must be mathematically converted to delta-equivalent equity positions ($\Delta_{position} = \frac{\partial V}{\partial S} \times Multiplier$) before Fama-French regression.