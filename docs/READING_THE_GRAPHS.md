# Reading the Manager DNA Outputs

A guide to interpreting every plot, CSV, and graph file the pipeline produces. Each section answers: *what is this, what does it tell me, and what should I flag?*

---

## 1. `market_regimes_gmm.png` — Regime Classification

**What it shows**
SPY's cumulative return curve, with every trading day colored by its GMM-assigned regime (green = 0, red = 1, orange = 2; ordering varies per run).

**How to read it**
- **Color clustering in time** = the regime captures a temporally coherent market state (e.g., red blob during a drawdown). This is good.
- **Color salt-and-pepper** = the regime is splitting noise, not state. Flag.
- Vertical color transitions during well-known events (COVID March 2020, Oct 2022 inflation peak, 2023 banking stress) are external sanity checks.

**Pair with** `get_regime_summary()` table — match each color to its mean SPY / VIX / Credit / 10Y values to assign an economic label:
- High VIX + negative SPY mean → **Stress**
- Low VIX + positive SPY mean → **Bull-calm**
- Rising 10Y + wide credit → **Rising-rate / late-cycle**

**Watch for**
- A regime with `N_Days < 30` — too thin for reliable beta averaging.
- A regime whose mean SPY return is positive but the color hits all the major drawdowns — GMM labels are nominal; reorder logic by feature, not by index.

---

## 2. `managerial_dna_network.png` — Bipartite Conviction Network

**What it shows**
A two-column graph. **Left column** = fund-regime nodes (e.g., `ARKK_R0`, light blue circles). **Right column** = Super-Style PCs (light green squares). An edge connects a fund-regime to a PC if its absolute PCA loading exceeds `edge_threshold` (default 0.5).

**Edge encoding**
| Visual | Meaning |
|---|---|
| **Blue solid** | Fund loads *positively* on that PC (long tilt) |
| **Red dashed** | Fund loads *negatively* on that PC (short / underweight tilt) |
| **Thick edge** | High |loading| = high conviction |
| **Thin edge** | Marginal exposure |
| **No edge** | Below threshold — neutral on that style |

**How to read it**
1. **Find each fund's three nodes** (R0, R1, R2). If all three connect to the **same PC with the same sign**, the manager is stylistically consistent across regimes.
2. **If a fund's nodes connect to different PCs in different regimes**, you've found *style drift* — the manager rotates exposures with the market state.
3. **Funds with no edges at all** are uninterpretable at this threshold — drop the threshold or re-examine.

**Watch for**
- A single fund (e.g., ARKK) connecting to a PC alone with very thick edges — that PC is mostly defined *by* that fund. Confirm via the drop-one sensitivity CSV.
- All funds connecting to PC1 with the same sign — your PCs aren't capturing dispersion; widen the universe.

**Interpreting PC meaning**: read the row of `super_styles` printed alongside. e.g., if PC1 = (+0.54 Mkt-RF, +0.34 SMB, −0.50 RMW, −0.44 CMA), then PC1 = "Aggressive Growth ↔ Defensive Quality-Value." Edges *into* PC1 inherit that semantics.

---

## 3. Rolling FF Beta Plot (Cell 9 in the notebook)

**What it shows**
Five time series for one fund (Mkt-RF, SMB, HML, RMW, CMA) over the full sample, each a rolling 63-day OLS beta of `fund - benchmark` returns on that factor.

**How to read it**
- **Stable horizontal lines** = rules-based / passive exposure. Expected for SCHD, VTV, MTUM.
- **Trending betas** = the manager is changing exposures over time. Expected for discretionary active funds.
- **Spikes around known events** = the manager reacted to that event. Cross-reference with the regime plot — does the spike fall in a stress regime?
- **One beta near ±1 with the others near 0** = the fund is essentially a single-factor proxy.

**Watch for**
- Betas swinging through zero — the fund flips long↔short on that factor. Real signal for active funds, suspect for passive ones (check data).
- Betas with R² < 0.3 (not plotted here, but a regression diagnostic) — the FF model isn't explaining the fund's variance.

---

## 4. `regime_layer_R{0,1,2}.graphml` — Multilayer Regime Networks

**What they are**
Three separate graphs, one per regime. Each has 5 nodes (the funds, no regime suffix). An edge between two funds in layer R_r is the **absolute cosine similarity** of their 5-factor mean betas *within regime r*. Higher edge weight = the two funds are behaving similarly in that regime.

**How to read them (in Gephi or any GraphML viewer)**

1. Open all three layers side-by-side.
2. **Run the same layout on each** (ForceAtlas2 with identical parameters) so positions are comparable.
3. **Compare topologies across regimes**:
   - Same clusters in all 3 layers → universe structure is *regime-invariant*.
   - Cluster membership shuffles between layers → regime-dependent peer groups; this is exactly what the project calls "Dynamic 4D Style Box."
4. **Look for bridges** — funds whose role changes from peripheral in one regime to central in another. These are the tactical managers.

**Pair with** `regime_spectral_distance.csv`. A large spectral distance between two layers numerically confirms what the visual side-by-side shows.

**Watch for**
- All edge weights near 1.0 — the cosine similarity is saturating because all funds have similar factor signs. Switch to signed similarity or PCA-projected coordinates.
- Disconnected nodes — that fund has no above-zero similarity with any peer in that regime; usually means an outlier (e.g., ARKK in this universe).

---

## 5. CSV Outputs

### `fund_communities.csv`
Columns: `node, fund, regime, community`.
- **community** = Louvain cluster id (0 is the largest).
- **A fund whose 3 rows all have the same `community` value** = stylistically stable across regimes.
- **A fund whose 3 rows span different communities** = the manager's peer group depends on regime — true style rotation.

### `fund_style_consistency.csv`
One row per fund, column `communities_spanned`.
- `1` = the fund clusters with the same peer group in all three regimes (stable DNA).
- `2` or `3` = the fund changes peer groups across regimes. The higher the number, the more tactical.

This is the headline number to report for each fund.

### `regime_spectral_distance.csv`
Columns: `regime_a, regime_b, spectral_distance` for each unordered pair of regimes.
- **Spectral distance ≈ 0** between two regimes = the universe's similarity topology is preserved between those market states.
- **Large spectral distance** = the network *reorganizes* between those regimes.
- The largest pair tells you which regime transition matters most for the universe.

There is no universal threshold; report ratios. If R0↔R1 is 0.4 and R0↔R2 is 1.8, the universe rearranges 4.5× more between R0 and R2 than between R0 and R1.

### `within_fund_drift.csv`
Per-fund standard deviation of mean betas across regimes, for each of the 5 FF factors, plus `Total_Drift`.
- **Total_Drift** ranks funds from most regime-sensitive (top) to most static (bottom).
- High values in `HML` or `RMW` columns alone tell you *which* style the manager rotates — useful for narrative.

### `pca_sensitivity_drop_one.csv`
For each dropped fund, the cosine similarity between PC1 of the reduced model and PC1 of the full model.
- **`pc1_cosine_vs_full` ≥ 0.95** = the basis is stable without that fund. Good.
- **`pc1_cosine_vs_full` ≤ 0.80** = that fund is anchoring PC1; the result is fragile.
- Bonus: compare `pc1_var_explained` across rows. If dropping a fund causes PC1 explained variance to fall from 65% to 40%, that fund was carrying the signal.

### `pca_bootstrap_variance.csv`
Mean, std, p05, p50, p95 of each PC's explained variance ratio over 1,000 bootstrap resamples of the rows.
- **Tight bands** (p95 − p05 < 0.05) = the variance-explained estimate is stable.
- **Wide bands** = small-sample noise dominates. With n=15 expect ±5–10pp on PC1.
- Report PC1 as "65% ± [p05, p95]" rather than as a point estimate.

### `style_drift_classification.csv`
Per-fund classification combining the three drift signals into one label.
Columns: `total_drift, communities_spanned, universe_spectral_max, drift_class`.
- **drift_class = "Drifting"** — fund has high cross-regime drift *and* its peer
  group changes (communities_spanned > 1). Strong active-management signal.
- **drift_class = "Rotational"** — one of the two conditions holds. Modest signal.
- **drift_class = "Stable"** — neither. Likely a passive / rules-based fund.

This is the **single headline label** to attach to each fund. The other CSVs
provide the supporting evidence.

### `managerial_dna_network.html`
Interactive Plotly version of the bipartite network plot. Open in any browser.
- **Hover a fund node** → tooltip with that fund-regime's PCA loadings on each PC.
- **Hover a Super-Style node** → tooltip with the PC's loadings on each FF factor.
- **Hover an edge** → tooltip with weight and sign.
- Edge color and dashing match the static PNG (blue solid = long, red dashed = short).

Enabled by `dna_mapper.interactive_html: true` in `config.yaml`. Requires
`pip install plotly` (or `pip install -e ".[interactive]"`).

### Hull-adjusted `{TICKER}_factor_loadings.csv`
When `hull_adjustment` is configured for a fund, its factor loadings CSV is
saved post-adjustment. The DataFrame's `attrs` dict records the multiplier
applied (`hull_multiplier`). A multiplier of 1.00 means no adjustment; 1.15
means options exposure increased effective equity beta by 15%; 0.85 means
options reduced it by 15% (typical for covered-call strategies).

Populate the `hull_adjustment` block in `config.yaml` from SEC Form N-PORT
disclosures or fund company holdings reports. See `hull_adjustment.py` for
the option position schema.

### `regime_labels.csv` and `{TICKER}_factor_loadings.csv`
Raw per-day outputs. Useful for custom analyses but not the headline artifacts.

---

## Workflow: Reading a Run End-to-End

1. **Sanity-check the regime plot** — colors should follow market state, not random.
2. **Read the regime summary table** — assign economic labels (stress / bull / rate-up). Check `N_Days` is healthy in each.
3. **Open `pca_sensitivity_drop_one.csv` first.** If PC1 falls apart without one fund, the rest of the analysis is fragile — stop and expand the universe.
4. **Open `pca_bootstrap_variance.csv`** — get honest CIs for the variance explained.
5. **Look at `fund_style_consistency.csv`** — the headline behavioral claim. Which funds drift, which don't?
6. **Cross-check with the bipartite network plot** — visual confirmation of the community structure.
7. **For each "drifting" fund, read `within_fund_drift.csv`** — *which* factor is rotating?
8. **For universe-level drift, read `regime_spectral_distance.csv`** — does the whole topology rearrange, or do only individual funds move?
9. **Open the three `.graphml` layers in Gephi** for the visual story.

---

## Common Gotchas

- **GMM regime labels (0/1/2) are nominal and can reorder between runs.** Always map them to economic states via the summary table before comparing across runs.
- **The bipartite plot's edge threshold is arbitrary.** Trust `fund_communities.csv` (no threshold needed) over the visual.
- **PCA signs are arbitrary.** PC1 = +Mkt-RF or −Mkt-RF carries the same information; don't over-interpret the direction.
- **Cosine similarity ignores magnitude.** Two funds with identical factor *profile* but different *aggressiveness* will look identical in the regime layers. For magnitude-sensitive analysis, switch to the raw Euclidean distance.
- **n=15 is small.** Always read PCA results through the bootstrap and drop-one CSVs, not as point estimates.
