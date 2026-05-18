"""
Stage 3: PCA & Bipartite Network Clustering

Extracts orthogonal "Super-Styles" via PCA over FF factor space, then builds a
bipartite graph G = (Funds, Super-Styles, Edges) for managerial DNA clustering.

References:
    - Stanford: "Classification of Assets and Investors in a Financial Bipartite Graph"
    - Hull: Delta-equivalent position adjustment (future work)
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class ManagerialDNAMapper:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.super_styles = None
        self.fund_loadings = None

    def extract_super_styles(self, aggregated_factor_data):
        print(f"Running PCA to extract top {self.n_components} Super-Styles...")

        scaled_data = self.scaler.fit_transform(aggregated_factor_data)
        self.pca.fit(scaled_data)

        fund_pca_loadings = self.pca.transform(scaled_data)
        self.fund_loadings = pd.DataFrame(
            fund_pca_loadings,
            index=aggregated_factor_data.index,
            columns=[f"Style_PC{i+1}" for i in range(self.n_components)],
        )

        self.super_styles = pd.DataFrame(
            self.pca.components_,
            columns=aggregated_factor_data.columns,
            index=[f"Style_PC{i+1}" for i in range(self.n_components)],
        )

        print(f"Variance explained: {self.pca.explained_variance_ratio_}")
        return self.fund_loadings, self.super_styles

    def sensitivity_drop_one(self, aggregated_factor_data):
        """Refit PCA dropping all rows for each fund; report PC1 cosine vs full fit."""
        full = PCA(n_components=self.n_components)
        full.fit(StandardScaler().fit_transform(aggregated_factor_data))
        pc1_full = full.components_[0]

        funds = sorted({idx.rsplit("_R", 1)[0] for idx in aggregated_factor_data.index})
        rows = []
        for fund in funds:
            mask = ~aggregated_factor_data.index.str.startswith(f"{fund}_R")
            subset = aggregated_factor_data[mask]
            if len(subset) < self.n_components + 1:
                continue
            p = PCA(n_components=self.n_components)
            p.fit(StandardScaler().fit_transform(subset))
            pc1_sub = p.components_[0]
            cos = float(np.dot(pc1_full, pc1_sub) /
                        (np.linalg.norm(pc1_full) * np.linalg.norm(pc1_sub)))
            rows.append({
                "dropped_fund": fund,
                "pc1_cosine_vs_full": abs(cos),
                "pc1_var_explained": float(p.explained_variance_ratio_[0]),
                "pc2_var_explained": float(p.explained_variance_ratio_[1]) if self.n_components > 1 else np.nan,
            })
        return pd.DataFrame(rows).set_index("dropped_fund")

    def fund_projection_communities(self, similarity_threshold=0.0, seed=42):
        """One-mode projection of bipartite graph onto fund-regime nodes via cosine
        similarity of PCA loadings, then Louvain community detection.

        Closet-indexing test: a fund whose entire (Fund_R0, Fund_R1, Fund_R2) trio
        lands in the same community as a benchmark proxy is statistically
        indistinguishable from passive exposure.
        """
        if self.fund_loadings is None:
            raise RuntimeError("Call extract_super_styles first.")

        L = self.fund_loadings.values
        norms = np.linalg.norm(L, axis=1, keepdims=True)
        L_unit = L / np.where(norms == 0, 1, norms)
        S = L_unit @ L_unit.T
        np.fill_diagonal(S, 0.0)

        nodes = self.fund_loadings.index.tolist()
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                w = float(S[i, j])
                if w > similarity_threshold:
                    G.add_edge(nodes[i], nodes[j], weight=w)

        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, weight="weight", seed=seed)
        except ImportError:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G, weight="weight"))

        rows = []
        for cid, comm in enumerate(sorted(communities, key=lambda c: -len(c))):
            for node in comm:
                fund, _, regime = node.rpartition("_R")
                rows.append({"node": node, "fund": fund, "regime": int(regime), "community": cid})
        df = pd.DataFrame(rows).set_index("node").sort_values(["community", "fund", "regime"])

        # Fund-level consistency: how many distinct communities does each fund's
        # 3 regime nodes span? 1 = consistent style, >1 = regime-dependent style.
        consistency = df.groupby("fund")["community"].nunique().rename("communities_spanned")
        return df, G, consistency

    def regime_spectral_distance(self, aggregated_factor_data, factor_cols):
        """Build a fund-fund similarity graph per regime layer and compute pairwise
        Laplacian spectral distance between layers.

        Mathematically: L_r = D_r - A_r where A_r is the |cosine| similarity of
        fund factor vectors within regime r. Spectral distance between layers
        r and r' = ||sort(eig(L_r)) - sort(eig(L_r'))||_2.

        Small spectral distance => the universe's *relative* style structure is
        preserved across regimes (no aggregate drift). Large => the network
        topology itself reorganizes between regimes.
        """
        regimes = sorted({int(idx.rsplit("_R", 1)[1]) for idx in aggregated_factor_data.index})
        layers = {}
        spectra = {}

        for r in regimes:
            mask = [idx for idx in aggregated_factor_data.index if idx.endswith(f"_R{r}")]
            sub = aggregated_factor_data.loc[mask].copy()
            sub.index = [idx.rsplit("_R", 1)[0] for idx in sub.index]

            X = sub[factor_cols].values
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            Xu = X / np.where(norms == 0, 1, norms)
            S = np.abs(Xu @ Xu.T)
            np.fill_diagonal(S, 0.0)

            D = np.diag(S.sum(axis=1))
            Lap = D - S
            eigs = np.sort(np.linalg.eigvalsh(Lap))

            G = nx.Graph()
            for i, u in enumerate(sub.index):
                G.add_node(u)
                for j in range(i + 1, len(sub.index)):
                    if S[i, j] > 0:
                        G.add_edge(u, sub.index[j], weight=float(S[i, j]))
            layers[r] = G
            spectra[r] = eigs

        rows = []
        rs = sorted(spectra)
        for i, r1 in enumerate(rs):
            for r2 in rs[i + 1:]:
                d = float(np.linalg.norm(spectra[r1] - spectra[r2]))
                rows.append({"regime_a": r1, "regime_b": r2, "spectral_distance": d})
        return pd.DataFrame(rows), layers, spectra

    def bootstrap_pca(self, aggregated_factor_data, n_iter=1000, random_state=42):
        """Bootstrap rows with replacement; return distribution of variance explained per PC."""
        rng = np.random.default_rng(random_state)
        n = len(aggregated_factor_data)
        records = np.zeros((n_iter, self.n_components))
        for i in range(n_iter):
            idx = rng.integers(0, n, size=n)
            sample = aggregated_factor_data.iloc[idx]
            try:
                p = PCA(n_components=self.n_components)
                p.fit(StandardScaler().fit_transform(sample))
                records[i] = p.explained_variance_ratio_
            except Exception:
                records[i] = np.nan
        df = pd.DataFrame(records, columns=[f"PC{i+1}" for i in range(self.n_components)])
        summary = pd.DataFrame({
            "mean": df.mean(),
            "std": df.std(),
            "p05": df.quantile(0.05),
            "p50": df.quantile(0.50),
            "p95": df.quantile(0.95),
        })
        return summary

    def build_bipartite_network(self, edge_threshold=0.5):
        print("Constructing bipartite DNA network...")
        B = nx.Graph()

        funds = self.fund_loadings.index.tolist()
        styles = self.fund_loadings.columns.tolist()

        B.add_nodes_from(funds, bipartite=0)
        B.add_nodes_from(styles, bipartite=1)

        for fund in funds:
            for style in styles:
                weight = self.fund_loadings.loc[fund, style]
                if abs(weight) > edge_threshold:
                    B.add_edge(fund, style, weight=abs(weight), sign=np.sign(weight))

        return B

    def visualize_network(self, B, save_path="output/managerial_dna_network.png"):
        plt.figure(figsize=(12, 8))

        top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
        pos = nx.bipartite_layout(B, top_nodes)

        nx.draw_networkx_nodes(B, pos, nodelist=top_nodes,
                               node_color="lightblue", node_size=2000, label="Active ETFs")
        nx.draw_networkx_nodes(B, pos, nodelist=set(B) - top_nodes,
                               node_color="lightgreen", node_size=3000, node_shape="s",
                               label="Super-Styles (PCs)")

        edges = B.edges(data=True)
        pos_edges = [(u, v) for u, v, d in edges if d["sign"] > 0]
        neg_edges = [(u, v) for u, v, d in edges if d["sign"] < 0]
        w_pos = [d["weight"] * 2 for u, v, d in edges if d["sign"] > 0]
        w_neg = [d["weight"] * 2 for u, v, d in edges if d["sign"] < 0]

        nx.draw_networkx_edges(B, pos, edgelist=pos_edges, width=w_pos,
                               edge_color="blue", alpha=0.6)
        nx.draw_networkx_edges(B, pos, edgelist=neg_edges, width=w_neg,
                               edge_color="red", alpha=0.6, style="dashed")

        nx.draw_networkx_labels(B, pos, font_size=10, font_weight="bold")

        plt.title("Managerial DNA: Active ETF Bipartite Conviction Network")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Network plot saved: {save_path}")
        plt.close()
