# Manager DNA

Reverse-engineer the behavioral "DNA" of active ETF managers. Answers: *does this fund manager have genuine alpha, or are they just taking on disguised factor risk?*

Output is a **Dynamic 4D Style Box** — a regime-aware, mathematically rigorous alternative to the static Morningstar 3×3 grid.

Presented as a Math & CS capstone combining statistical learning (Fama-French OLS, GMM, PCA), discrete math (bipartite graphs, Louvain modularity, spectral distance), and financial econometrics.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/james-kidd/manager-dna.git
cd manager-dna

# 2. Install (editable mode — changes to src/ take effect immediately)
pip install -e .

# 3. Run the full pipeline
python run_pipeline.py
```

Outputs land in `output/` — plots, CSVs, and GraphML files.

### Optional extras

```bash
pip install -e ".[notebook]"       # Jupyter + ipywidgets
pip install -e ".[interactive]"    # Plotly (interactive HTML network)
```

### Google Colab

Open [`notebooks/manager_dna_colab.ipynb`](notebooks/manager_dna_colab.ipynb) — it handles `git clone` and `pip install` automatically and mounts Google Drive for persistent outputs.

---

## CLI Options

```bash
python run_pipeline.py                              # use config.yaml defaults
python run_pipeline.py --fund_ticker QQQ            # override single fund
python run_pipeline.py --n_regimes 4                # override GMM regime count
python run_pipeline.py --n_components 3             # override PCA components
python run_pipeline.py --config my_config.yaml      # custom config file
```

---

## Configuration

All tunable parameters live in [`config.yaml`](config.yaml). Top-level sections:

| Section | Controls |
|---|---|
| `factor_extraction` | Benchmark ticker, start date, rolling OLS window |
| `regime_model` | Number of GMM regimes, start date |
| `dna_mapper` | PCA components, edge threshold, bootstrap iterations, drift quantile, interactive HTML toggle |
| `fund_universe` | List of ETF tickers to analyze (13 by default) |
| `hull_adjustment` | Per-fund options-exposure spec for delta-equivalent beta adjustment (empty by default — populate from SEC N-PORT) |
| `output` | Output directory and filenames |

---

## What the Pipeline Does

Three stages, orchestrated by `pipeline.py`:

1. **Factor Extraction** — Rolling OLS of `fund - benchmark` against the Fama-French 5-factor model (Mkt-RF, SMB, HML, RMW, CMA). Default window: 63 trading days (~1 quarter).
2. **Regime Modeling** — Gaussian Mixture Model over four macro features (SPY return, VIX, HY credit spread, 10Y yield change) classifies every trading day into K regimes (default K=3).
3. **DNA Mapping** — Aggregates factor loadings per (fund × regime), runs PCA to extract "Super-Styles," and constructs a bipartite graph of funds ↔ super-styles plus a per-regime fund-fund similarity layer.

Plus six diagnostic / analytical CSVs covering:
- Drop-one PCA sensitivity (basis stability)
- Bootstrap variance-explained CIs
- Louvain communities (data-driven peer groups)
- Per-regime spectral distance (universe-level drift)
- Within-fund drift (per-fund regime sensitivity)
- Style drift classification (Stable / Rotational / Drifting)

See [`docs/READING_THE_GRAPHS.md`](docs/READING_THE_GRAPHS.md) for a full reader's guide.

---

## Project Structure

```
manager-dna/
├── src/manager_dna/           # Installable Python package
│   ├── factor_extraction.py   # Stage 1: Rolling FF OLS
│   ├── regime_model.py        # Stage 2: GMM regime classification
│   ├── dna_mapper.py          # Stage 3: PCA + bipartite + diagnostics
│   ├── hull_adjustment.py     # Delta-equivalent beta adjustment for options
│   └── pipeline.py            # Unified three-stage orchestrator
├── prototype_py/              # Original standalone scripts (preserved)
├── skill_manager_dna/         # Research docs & math reference
├── docs/READING_THE_GRAPHS.md # Reader's guide for every output
├── notebooks/
│   └── manager_dna_colab.ipynb
├── output/                    # Generated plots & CSVs (gitignored)
├── config.yaml                # All tunable parameters
├── run_pipeline.py            # CLI entry point
└── pyproject.toml
```

---

## Outputs

Running the pipeline produces (in `output/` or your configured directory):

**Plots**
- `market_regimes_gmm.png` — SPY cumulative return colored by regime
- `managerial_dna_network.png` — Static bipartite network
- `managerial_dna_network.html` — Interactive Plotly version (if `interactive_html: true`)

**CSVs**
- `{TICKER}_factor_loadings.csv` — Rolling FF betas per fund
- `regime_labels.csv` — Regime classification per trading day
- `within_fund_drift.csv` — Per-fund std of regime-mean betas
- `pca_sensitivity_drop_one.csv` — PC1 stability when each fund is dropped
- `pca_bootstrap_variance.csv` — Bootstrap CIs on variance explained
- `fund_communities.csv` — Louvain community assignment per fund-regime
- `fund_style_consistency.csv` — Communities spanned per fund across regimes
- `regime_spectral_distance.csv` — Pairwise Laplacian spectral distance between regime layers
- `style_drift_classification.csv` — Per-fund label: Stable / Rotational / Drifting

**Graph data**
- `regime_layer_R{0,1,2}.graphml` — Per-regime fund-fund similarity graphs (open in Gephi)

---

## Key Domain Concepts

- **Active beta**: factor loading from `fund_return - benchmark_return`. Isolates managerial skill from passive market exposure.
- **Super-Style**: a PCA principal component over the FF factor space. A latent behavioral archetype.
- **Style drift**: substantial change in a fund's bipartite-graph topology or community membership between GMM regimes.
- **Hull constraint**: delta-equivalent adjustment for options-holding funds. `Δ_eq = Δ × Multiplier × Contracts × Spot`.

---

## Dependencies

Defined in `pyproject.toml` and `requirements.txt`. Core stack:

```
pandas numpy yfinance pandas-datareader statsmodels scikit-learn networkx matplotlib pyyaml
```

Optional: `jupyter ipywidgets` (notebook), `plotly` (interactive viz).

Python 3.10+.

---

## References

- Fama, E. F. & French, K. R. (2015). *A five-factor asset pricing model.* Journal of Financial Economics.
- Cremers, M. & Petajisto, A. (2009). *How active is your fund manager? A new measure that predicts performance.* RFS.
- Hull, J. C. *Options, Futures, and Other Derivatives*, Ch. 19.
- Mantegna, R. N. (1999). *Hierarchical structure in financial markets.* European Physical Journal B.
- Two Sigma (2017). *A Machine Learning Approach to Regime Modeling.*
- Stanford CS224W lecture notes — Bipartite graphs in financial networks.

---

## License

MIT. See `pyproject.toml`.
