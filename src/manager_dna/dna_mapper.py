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
