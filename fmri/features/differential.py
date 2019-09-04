import numpy as np
from scipy.linalg import expm, fractional_matrix_power as powm

import networkx as nx

class DifferentialFeatureExtractor():

    def hks(self, t=0.1, scale_invariant=False):
        L_normalized = np.array(nx.normalized_laplacian_matrix(self.G).todense())
        H = np.real(expm(-t * L_normalized))
        hks_t = np.diagonal(H)

        return hks_t

    def wks(self, t=0.1):
        L_normalized = np.array(nx.normalized_laplacian_matrix(self.G).todense())
        W = np.real(expm(-1j * t * L_normalized))
        wks_t = np.diagonal(W)
        
        return wks_t
