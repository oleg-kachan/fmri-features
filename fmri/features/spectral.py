import numpy as np
from scipy.linalg import fractional_matrix_power as powm

import networkx as nx
#from fmri.features.sunbeam import nbvals

class SpectralFeatureExtractor():

    def spectral_matrix(self):
        l, _ = np.linalg.eigh(self.A[0])
        return l

    def spectral_laplacian(self, normalized=False):

        # normalized Laplacian
        if normalized:
            # deprecated as NetworkX implementation is more stable
            #A = nx.to_numpy_array(self.G) 
            #D = np.diag(A.sum(axis=1))
            #D_pow = powm(D, -0.5)
            #L = D - A
            #L_normalized = D_pow @ L @ D_pow

            G = nx.from_numpy_array(self.A[0])
            L_normalized = np.array(nx.normalized_laplacian_matrix(G).todense())
            l, _ = np.linalg.eigh(L_normalized)

        # Laplacian
        else:
            D = np.diag(self.A[0].sum(axis=1))
            L = D - self.A[0]
            l, _ = np.linalg.eigh(L)

        return l

    def spectral_poly(self, k_max=3):
        polynomials = []

        for k in range(2, k_max+1):
            polynomial_k = np.trace(powm(self.A[0], k))
            polynomials.append(polynomial_k)

        return np.array(polynomials)
    
    def spectral_non_backtracking(self, topk="automatic"):
        return nbvals(self.G, topk=topk, fmt="1D")
