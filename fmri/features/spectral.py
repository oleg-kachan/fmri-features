import numpy as np
from scipy.linalg import fractional_matrix_power as powm

import networkx as nx
from fmri.features.sunbeam import nbvals

class SpectralFeatureExtractor():

    def spectral_adjacency(self):
        A = nx.to_numpy_array(self.G)
        l, _ = np.linalg.eigh(A)
        return l

    def spectral_laplacian(self):
        A = nx.to_numpy_array(self.G)
        D = np.diag(A.sum(axis=1))
        L = D - A
        l, _ = np.linalg.eigh(L)
        return l

    def spectral_laplacian_normalized(self):
        #A = nx.to_numpy_array(self.G)
        #D = np.diag(A.sum(axis=1))
        #D_pow = powm(D, -0.5)
        #L = D - A
        #L_normalized = D_pow @ L @ D_pow
        L_normalized = np.array(nx.normalized_laplacian_matrix(self.G).todense())
        l, _ = np.linalg.eigh(L_normalized)
        return l
    
    def spectral_non_backtracking(self, topk="automatic"):
        return nbvals(self.G, topk=topk, fmt="1D")
