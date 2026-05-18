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
        """
        Initialize the Mapper.
        n_components determines how many "Super-Styles" we want to extract from the 
        Fama-French factor space. Typically, the first 3 PCs explain >80% of variance.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.scaler = StandardScaler()
        self.super_styles = None
        self.fund_loadings = None
        
    def extract_super_styles(self, aggregated_factor_data):
        """
        Step 1: Principal Component Analysis
        aggregated_factor_data: A DataFrame where index is Funds, columns are average 
        Fama-French Active Bet betas over a specific Regime.
        """
        print(f"Running PCA to extract Top {self.n_components} Super-Styles...")
        
        # Standardize the betas before running PCA to prevent magnitude bias
        scaled_data = self.scaler.fit_transform(aggregated_factor_data)
        
        # Fit PCA
        self.pca.fit(scaled_data)
        
        # Transform the funds into the new PCA space (These are the Fund's loadings on the Super-Styles)
        fund_pca_loadings = self.pca.transform(scaled_data)
        self.fund_loadings = pd.DataFrame(fund_pca_loadings, 
                                          index=aggregated_factor_data.index, 
                                          columns=[f'Style_PC{i+1}' for i in range(self.n_components)])
        
        # View what the Super-Styles actually represent in terms of Fama-French
        self.super_styles = pd.DataFrame(self.pca.components_, 
                                         columns=aggregated_factor_data.columns, 
                                         index=[f'Style_PC{i+1}' for i in range(self.n_components)])
        
        print("PCA Extraction Complete. Variance Explained:", self.pca.explained_variance_ratio_)
        return self.fund_loadings, self.super_styles

    def build_bipartite_network(self, edge_threshold=0.5):
        """
        Step 2: Network Graphing (Stanford Methodology)
        Builds a bipartite graph connecting Funds (Investors) to Super-Styles (Assets/Concepts).
        edge_threshold filters out weak connections to clean up the graph.
        """
        print("Constructing Bipartite DNA Network...")
        B = nx.Graph()
        
        funds = self.fund_loadings.index.tolist()
        styles = self.fund_loadings.columns.tolist()
        
        # Add Nodes with bipartite attribute
        B.add_nodes_from(funds, bipartite=0) # Node Set 1: Funds
        B.add_nodes_from(styles, bipartite=1) # Node Set 2: Styles
        
        # Add Edges based on PCA loadings
        for fund in funds:
            for style in styles:
                weight = self.fund_loadings.loc[fund, style]
                # Only add an edge if the fund has a strong conviction (loading) in this style
                if abs(weight) > edge_threshold:
                    # We use absolute weight for edge thickness, but keep the sign for color (Long/Short style)
                    B.add_edge(fund, style, weight=abs(weight), sign=np.sign(weight))
                    
        return B

    def visualize_network(self, B):
        """
        Visualize the resulting managerial clusters.
        """
        plt.figure(figsize=(12, 8))
        
        # Define the layout for a Bipartite Graph
        top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
        pos = nx.bipartite_layout(B, top_nodes)
        
        # Node styling
        nx.draw_networkx_nodes(B, pos, nodelist=top_nodes, node_color='lightblue', node_size=2000, label="Active ETFs")
        nx.draw_networkx_nodes(B, pos, nodelist=set(B) - top_nodes, node_color='lightgreen', node_size=3000, node_shape='s', label="Super-Styles (PCs)")
        
        # Edge styling based on conviction (weight) and direction (sign)
        edges = B.edges(data=True)
        positive_edges = [(u, v) for u, v, d in edges if d['sign'] > 0]
        negative_edges = [(u, v) for u, v, d in edges if d['sign'] < 0]
        weights_pos = [d['weight'] * 2 for u, v, d in edges if d['sign'] > 0]
        weights_neg = [d['weight'] * 2 for u, v, d in edges if d['sign'] < 0]
        
        nx.draw_networkx_edges(B, pos, edgelist=positive_edges, width=weights_pos, edge_color='blue', alpha=0.6)
        nx.draw_networkx_edges(B, pos, edgelist=negative_edges, width=weights_neg, edge_color='red', alpha=0.6, style='dashed')
        
        nx.draw_networkx_labels(B, pos, font_size=10, font_weight='bold')
        
        plt.title('Managerial DNA: Active ETF Bipartite Conviction Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("managerial_dna_network.png")
        print("Network plot saved as managerial_dna_network.png")

# --- Execution Example ---
if __name__ == "__main__":
    # Simulated aggregated factor data for 5 ETFs over a specific regime (e.g., "Crisis Regime")
    # Columns represent active bets against: Market, Size, Value, Profitability, Investment
    simulated_data = {
        'Mkt-RF': [0.5, 0.6, -0.2, -0.1, 0.8],
        'SMB': [0.8, 0.7, 0.1, -0.5, 0.9],
        'HML': [-1.2, -1.0, 0.8, 0.9, -1.5],
        'RMW': [-0.5, -0.4, 0.7, 0.8, -0.6],
        'CMA': [-0.8, -0.7, 0.5, 0.6, -1.0]
    }
    # Funds: Two aggressive tech/growth funds, two conservative value funds, one extreme momentum fund
    funds = ['ARKK_Proxy', 'Growth_ETF', 'Value_ETF', 'Defensive_ETF', 'Momentum_ETF']
    df_regime_bets = pd.DataFrame(simulated_data, index=funds)
    
    mapper = ManagerialDNAMapper(n_components=2)
    fund_loadings, super_styles = mapper.extract_super_styles(df_regime_bets)
    
    print("\nSuper-Style Formations (What the PCs represent):")
    print(super_styles)
    
    print("\nManagerial DNA Loadings (Fund exposure to PCs):")
    print(fund_loadings)
    
    dna_network = mapper.build_bipartite_network(edge_threshold=0.5)
    mapper.visualize_network(dna_network)