# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Manager DNA** is a quantitative finance research tool that reverse-engineers the behavioral "DNA" of active ETF managers. It answers: *does this fund manager have genuine alpha, or are they just taking on disguised factor risk?* The output is a **Dynamic 4D Style Box** — a regime-aware, mathematically rigorous alternative to the static Morningstar 3×3 grid.

Presented as a Math & Computer Science capstone project combining statistical learning, network theory, and financial econometrics.

## Quick Start

```bash
# Install (editable mode — changes to src/ take effect immediately)
pip install -e ".[notebook]"

# Run the full pipeline
python run_pipeline.py

# Override parameters via CLI
python run_pipeline.py --fund_ticker QQQ --n_regimes 4

# Use a custom config
python run_pipeline.py --config my_config.yaml

# Google Colab
# Open notebooks/manager_dna_colab.ipynb — it handles clone + install automatically.
```

## Project Structure

```
manager-dna/
├── src/manager_dna/           # Installable Python package
│   ├── __init__.py
│   ├── factor_extraction.py   # Stage 1: Rolling FF OLS
│   ├── regime_model.py        # Stage 2: GMM regime classification
│   ├── dna_mapper.py          # Stage 3: PCA + bipartite network
│   └── pipeline.py            # Unified three-stage orchestrator
├── prototype_py/              # Original standalone scripts (preserved)
├── skill_manager_dna/         # Research docs & math reference
├── notebooks/
│   └── manager_dna_colab.ipynb
├── output/                    # Generated plots & CSVs (gitignored)
├── config.yaml                # All tunable parameters
├── run_pipeline.py            # CLI entry point
├── pyproject.toml             # Package metadata & deps
└── requirements.txt           # Flat dependency list
```

## Dependencies

Defined in `pyproject.toml` and `requirements.txt`:

```
pandas numpy yfinance pandas-datareader statsmodels scikit-learn networkx matplotlib pyyaml
```

Optional (for notebooks): `jupyter ipywidgets`

No test runner is configured yet.

## Architecture: Three-Stage Pipeline

All parameters are configurable via `config.yaml`. The pipeline runs Stages 2→1→3 (regime labels are needed to segment Stage 1 output).

### Stage 1 — `factor_extraction.py`
**Class:** `ManagerialFactorExtractor`

Rolling OLS of active return (`fund - benchmark`) against the Fama-French 5-Factor model. Window default: 63 trading days (≈1 quarter). Output: time-series of rolling `[alpha, Mkt-RF, SMB, HML, RMW, CMA]` betas per fund.

### Stage 2 — `regime_model.py`
**Class:** `MarketRegimeModel`

GMM over four macro features (SPY return, VIX, HY credit spread, 10Y yield change). Outputs regime labels (0/1/2) with confidence probabilities. GMM chosen over K-Means for soft probabilistic assignment on noisy financial data.

### Stage 3 — `dna_mapper.py`
**Class:** `ManagerialDNAMapper`

PCA extracts orthogonal "Super-Styles" from the aggregated fund×regime beta matrix. Bipartite graph `G = (Funds, Super-Styles, Edges)` visualizes conviction: blue = long tilt, red dashed = short tilt, edge width = loading magnitude.

### Pipeline — `pipeline.py`

Orchestrates all three stages:
1. Fits GMM to get regime labels
2. Runs FF extraction for each fund in `config.yaml → fund_universe`
3. Aggregates mean betas per (fund, regime) pair
4. Runs PCA + network on the aggregated matrix
5. Saves all outputs (CSVs + plots) to `output/`

## Key Domain Concepts

- **Active beta**: factor loading from `fund_return - benchmark_return`. Isolates managerial skill from passive market exposure.
- **Super-Style**: a PCA principal component over FF factor space. A latent behavioral archetype.
- **Style drift**: substantial change in a fund's bipartite graph topology between GMM regimes.
- **Hull Constraint**: delta-equivalent adjustment for options-holding funds. Formula: `Δ = ∂V/∂S × Multiplier`. Not yet implemented.

## Reference Documents (`skill_manager_dna/`)

- `financial_advisor_cfa.md` — Institutional context (hedge funds, wealth management, fund-of-funds)
- `mathematical_research_reference.md` — Formal math foundations with citations (Two Sigma, Stanford, Hull)
- `project_magement.md` — Original `ManagerialDNAMapper` prototype code
