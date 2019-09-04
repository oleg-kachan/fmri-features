import numpy as np
from scipy.linalg import fractional_matrix_power as powm

import networkx as nx

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

class GraphFeatureExtractor():

    def __init__(self, G, cache=False):
        self.G = G
        self.cache = cache

    def degree(self):
        return np.array(nx.degree(self.G, weight="weight"))[:,1]

    def betweenness_centrality(self):
        return np.array(list(nx.betweenness_centrality(self.G, weight="weight").values()))

    def closeness_centrality(self):
        return np.array(list(nx.closeness_centrality(self.G).values()))

    def eigenvector_centrality(self):
        return np.array(list(nx.eigenvector_centrality(self.G, weight="weight").values()))

    def second_order_centrality(self):
        return np.array(list(nx.second_order_centrality(self.G, weight="weight").values()))

    def clustering_coefficient(self):
        return np.array(list(nx.clustering(self.G, weight="weight").values()))

    def neighbor_degree(self):
        return np.array(list(nx.average_neighbor_degree(self.G, weight="weight").values()))

    def average_shortest_path(self):
        return nx.average_shortest_path_length(self.G, weight="weight")

    def global_efficiency(self):
        return nx.global_efficiency(self.G)

    def ollivier_ricci_curvature(self, alpha=0.5):
        return OllivierRicci(self.G, alpha=alpha, verbose="INFO").compute_ricci_curvature()

    def forman_ricci_curvature(self):
        return FormanRicci(self.G).compute_ricci_curvature()
