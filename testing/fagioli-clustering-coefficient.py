import networkx as nx
import numpy as np
import pandas as pd

# Create a small Weighted Directed Network (WDN)
# We'll define a graph with 4 nodes and directed, weighted edges
G = nx.DiGraph()

# Add weighted directed edges
edges = [
    (0, 1, 0.9),
    (1, 2, 0.8),
    (2, 0, 0.7),
    (0, 2, 0.5),
    (1, 0, 0.4),
    (2, 1, 0.3),
    (3, 0, 0.2),
    (1, 3, 0.6)
]

for u, v, w in edges:
    G.add_edge(u, v, weight=w)

A = nx.to_numpy_array(G, weight=None)
W = nx.to_numpy_array(G, weight='weight')

EYE = np.eye(G.number_of_nodes(), dtype=bool)

D_tot = (A + A.T).sum(axis=0)
D_bi = (A @ A)[EYE]
denominator = 2 * (D_tot * (D_tot - 1) - 2 * D_bi)

W_temp = (W ** (1/3)) + (W.T ** (1/3))
numerator = (W_temp @ W_temp @ W_temp)[EYE]

def get_clustering_coefficient(G):

    A = nx.to_numpy_array(G, weight=None)
    W = nx.to_numpy_array(G, weight='weight')

    EYE = np.eye(G.number_of_nodes(), dtype=bool)

    D_tot = (A + A.T).sum(axis=0)
    D_bi = (A @ A)[EYE]
    denominator = 2 * (D_tot * (D_tot - 1) - 2 * D_bi)

    W_temp = (W ** (1/3)) + (W.T ** (1/3))
    numerator = (W_temp @ W_temp @ W_temp)[EYE]

    return numerator / denominator