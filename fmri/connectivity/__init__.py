import numpy as np
from scipy.signal import detrend as dt

import fmri.connectivity.symmetric
import fmri.connectivity.asymmetric

class Connectivity(symmetric.SymmetricConnectivity, asymmetric.AsymmetricConnectivity):
    
    def __init__(self, ts, detrend=False):

        if len(ts) != 0:
            m, n = ts.shape[0], ts.shape[2]

            if detrend:
                for i in range(m):
                    for j in range(n):
                        ts[i,:,j] = dt(ts[i,:,j])

        self.ts = ts

    def get_thresholds(self, m=9):
        threshold = []
        for i in range(m):
            threshold.append((i + 1) / (m + 1))

        return threshold

    def threshold(self, R, threshold=None, weighted=False, reverse=False, abs=False):
        
        if isinstance(threshold, int):
            m = threshold
            threshold = []
            for i in range(m):
                threshold.append((i + 1) / (m + 1))

        R = R.copy()
        if abs:
            R = np.abs(R)

        if reverse:
            A = 1 - R
        else:
            A = R

        A_prime = A.copy()

        if reverse==False:

            # if threshold is specified
            if threshold is not None:

                # if a list of thresholds is specified, return a family of graphs
                if isinstance(threshold, list):
                    As = []
                    for threshold_i in threshold:
                        A_i = A.copy()
                        A_i[A_prime <= threshold_i] = 0

                        if not weighted:
                            A_i[A_prime > threshold_i] = 1
                
                        np.fill_diagonal(A_i, 0)

                        As.append(A_i)
                    
                # otherwise return single graph
                else:
                    A_prime = A.copy()
                    A[A_prime <= threshold] = 0

                    if not weighted:
                        A[A_prime > threshold] = 1
            
                    np.fill_diagonal(A, 0) # do not count self-loops

                    As = [A]

            # if threshold if not specified, return matrix as is
            else:
                As = [A]

        else: 
            # if threshold is specified
            if threshold is not None:

                # if a list of thresholds is specified, return a family of graphs
                if isinstance(threshold, list):
                    As = []
                    for threshold_i in threshold:
                        A_i = A.copy()
                        A_i[A_prime > threshold_i] = 0

                        if not weighted:
                            A_i[A_prime <= threshold_i] = 1
                
                        np.fill_diagonal(A_i, 0)

                        As.append(A_i)
                    
                # otherwise return single graph
                else:
                    A_prime = A.copy()
                    A[A_prime > threshold] = 0

                    if not weighted:
                        A[A_prime <= threshold] = 1
            
                    np.fill_diagonal(A, 0) # do not count self-loops

                    As = [A]

            # if threshold if not specified, return matrix as is
            else:
                As = [A]

        return np.array(As)