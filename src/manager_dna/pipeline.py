"""
Unified three-stage pipeline for Manager DNA analysis.

Stage 1 — Extract rolling Fama-French factor loadings per fund
Stage 2 — Classify market regimes via GMM
Stage 3 — PCA super-style extraction + bipartite network clustering

Usage:
    python run_pipeline.py                          # uses config.yaml defaults
    python run_pipeline.py --fund_ticker QQQ        # override single param
    python run_pipeline.py --config my_config.yaml  # custom config file
"""

import argparse
import os
import sys

import pandas as pd
import yaml

from .factor_extraction import ManagerialFactorExtractor, FF_FACTORS
from .regime_model import MarketRegimeModel
from .dna_mapper import ManagerialDNAMapper


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(cfg):
    out_dir = cfg.get("output", {}).get("dir", "output")
    os.makedirs(out_dir, exist_ok=True)

    # ── Stage 2: Regime modeling (run first — regime labels are needed to segment Stage 1 output) ──
    print("\n" + "=" * 60)
    print("STAGE 2: GMM Regime Modeling")
    print("=" * 60)

    rcfg = cfg["regime_model"]
    regime = MarketRegimeModel(
        start_date=rcfg["start_date"],
        end_date=rcfg.get("end_date"),
        n_regimes=rcfg["n_regimes"],
    )
    regime.fetch_macro_data()
    regime_data = regime.fit_predict_regimes()

    print("\nRegime characteristics (mean feature values):")
    print(regime.get_regime_summary())

    regime_plot = os.path.join(out_dir, cfg["output"].get("regime_plot", "market_regimes_gmm.png"))
    regime.plot_regimes(save_path=regime_plot)

    regime_csv = os.path.join(out_dir, cfg["output"].get("regime_csv", "regime_labels.csv"))
    regime_data[["Regime", "Regime_Probability"]].to_csv(regime_csv)
    print(f"Regime labels saved: {regime_csv}")

    # ── Stage 1: Factor extraction for each fund in the universe ──
    print("\n" + "=" * 60)
    print("STAGE 1: Fama-French Active Factor Extraction")
    print("=" * 60)

    fcfg = cfg["factor_extraction"]
    fund_universe = cfg.get("fund_universe", [fcfg["fund_ticker"]])

    all_loadings = {}
    for ticker in fund_universe:
        print(f"\n--- Processing {ticker} ---")
        extractor = ManagerialFactorExtractor(
            fund_ticker=ticker,
            benchmark_ticker=fcfg["benchmark_ticker"],
            start_date=fcfg["start_date"],
            window=fcfg["rolling_window"],
        )
        try:
            extractor.fetch_data()
            loadings = extractor.extract_rolling_factors()
            all_loadings[ticker] = loadings

            csv_path = os.path.join(out_dir, f"{ticker}_factor_loadings.csv")
            loadings.to_csv(csv_path)
            print(f"Saved: {csv_path}")
        except Exception as e:
            print(f"WARNING: Failed to process {ticker}: {e}")

    if not all_loadings:
        print("ERROR: No funds were successfully processed. Exiting.")
        sys.exit(1)

    # ── Combine: Aggregate factor loadings per regime ──
    print("\n" + "=" * 60)
    print("COMBINING: Aggregate betas by regime")
    print("=" * 60)

    regime_labels = regime_data["Regime"]
    aggregated_rows = []

    for ticker, loadings in all_loadings.items():
        common_dates = loadings.index.intersection(regime_labels.index)
        if len(common_dates) == 0:
            print(f"WARNING: No overlapping dates for {ticker}, skipping.")
            continue

        merged = loadings.loc[common_dates].copy()
        merged["Regime"] = regime_labels.loc[common_dates]

        for r in range(rcfg["n_regimes"]):
            regime_slice = merged[merged["Regime"] == r]
            if len(regime_slice) > 0:
                avg_betas = regime_slice[FF_FACTORS].mean()
                avg_betas.name = f"{ticker}_R{r}"
                aggregated_rows.append(avg_betas)

    if not aggregated_rows:
        print("ERROR: No aggregated data produced. Exiting.")
        sys.exit(1)

    df_agg = pd.DataFrame(aggregated_rows)
    print(f"\nAggregated factor matrix ({df_agg.shape[0]} fund-regime pairs x {df_agg.shape[1]} factors):")
    print(df_agg)

    # ── Stage 3: PCA + Network ──
    print("\n" + "=" * 60)
    print("STAGE 3: PCA Super-Styles & Bipartite Network")
    print("=" * 60)

    dcfg = cfg["dna_mapper"]
    n_comp = min(dcfg["n_components"], len(df_agg))
    mapper = ManagerialDNAMapper(n_components=n_comp)
    fund_loadings_pca, super_styles = mapper.extract_super_styles(df_agg)

    print("\nSuper-Style compositions (PC loadings on FF factors):")
    print(super_styles)
    print("\nFund-regime DNA loadings:")
    print(fund_loadings_pca)

    network = mapper.build_bipartite_network(edge_threshold=dcfg["edge_threshold"])

    net_plot = os.path.join(out_dir, cfg["output"].get("network_plot", "managerial_dna_network.png"))
    mapper.visualize_network(network, save_path=net_plot)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"All outputs saved to: {out_dir}/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Manager DNA Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--fund_ticker", help="Override primary fund ticker")
    parser.add_argument("--n_regimes", type=int, help="Override number of GMM regimes")
    parser.add_argument("--n_components", type=int, help="Override number of PCA components")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.fund_ticker:
        cfg["factor_extraction"]["fund_ticker"] = args.fund_ticker
        if args.fund_ticker not in cfg.get("fund_universe", []):
            cfg.setdefault("fund_universe", []).append(args.fund_ticker)
    if args.n_regimes:
        cfg["regime_model"]["n_regimes"] = args.n_regimes
    if args.n_components:
        cfg["dna_mapper"]["n_components"] = args.n_components

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
