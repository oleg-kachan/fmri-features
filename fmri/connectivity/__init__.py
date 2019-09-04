import numpy as np

import fmri.connectivity.correlation

class Connectivity(correlation.CorrelationConnectivity):
    
    def __init__(self, ts):
        self.ts = ts

    def threshold(self, R, threshold=0.3):
        R = R.copy()
        A = np.abs(R) # 1 - np.abs(R)
        A_prime = A.copy()
        A[A_prime < threshold] = 0
        #A[A_prime <= eps] = 1
        np.fill_diagonal(A, 0) # do not count self-loops
        return A