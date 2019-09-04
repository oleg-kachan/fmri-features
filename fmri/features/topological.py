import numpy as np
from scipy.linalg import fractional_matrix_power as powm

import networkx as nx

class TopologicalFeatureExtractor():

    def betti_curve(self, betti_number=0, shape=50):

        def threshold(A, eps):
            A = A.copy()
            A = 1 - np.abs(A)
            A_prime = A.copy()
            A[A_prime > eps] = 0
            A[A_prime <= eps] = 1
            np.fill_diagonal(A, 0) # do not count self-loops
            return A

        def betti_0(G):
            n_components = nx.number_connected_components(G)
            return n_components

        def betti_1(G):
            n_edges = G.number_of_edges()
            n_nodes = G.number_of_nodes()
            n_components = nx.number_connected_components(G)
            betti_1 = n_edges - n_nodes + n_components
            return betti_1

        if betti_number == 0:
            betti = betti_0
        elif betti_number == 1:
            betti = betti_1
        else:
            raise ValueError("Betti number should be 0 or 1")

        A = nx.to_numpy_array(self.G)

        betti_curve = []

        for eps in np.linspace(0, 1, num=shape):
            A_t = threshold(A, eps)
            G_t = nx.from_numpy_matrix(A_t)
            betti_curve.append(betti(G_t))
        
        return np.array(betti_curve)

    def persistence(self, k=[1], type="standard", representation="surface", shape=(16,16)):
        pass
