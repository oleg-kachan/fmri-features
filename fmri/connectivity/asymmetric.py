import numpy as np
import scipy as sc

from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
import statsmodels.tsa.stattools as tsa_tools

def absmax(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)

class AsymmetricConnectivity():

    def correlation_lagged(self, max_lag=10, estimator="maximum_likelihood", include_zero_shift=False, assume_centered=False):
        if estimator=="maximum_likelihood":
            correlation_measure = ConnectivityMeasure(kind="correlation", cov_estimator=EmpiricalCovariance(assume_centered=assume_centered))
        elif estimator=="ledoit_wolf":
            correlation_measure = ConnectivityMeasure(kind="correlation", cov_estimator=LedoitWolf(assume_centered=assume_centered))
        else:
            raise ValueError("Estimator should be 'maximum_likelihood' or 'ledoit_wolf'")
        
        # create connectivity matrix
        n, _, m = self.ts.shape
        R_lag = np.zeros((max_lag+1, n, m, m)) # max_lag + 1, to include_zero_shift if needed 

        for i in range(m):
            for j in range(m):
                for k in range(1, max_lag+1):
                    ts_knij = np.dstack((self.ts[:,k:,i], self.ts[:,:-k,j]))
                    R_lag[k-1,:,i,j] = np.nan_to_num(correlation_measure.fit_transform(ts_knij)[:,0,1])

                if include_zero_shift:
                    ts_knij = np.dstack((self.ts[:,:,i], self.ts[:,:,j]))
                    R_lag[max_lag,:,i,j] = np.nan_to_num(correlation_measure.fit_transform(ts_knij)[:,0,1])

        R_lag = absmax(R_lag, axis=0)

        return R_lag

    def granger(self, max_lag=10):
        
        # create connectivity matrix
        n, _, m = self.ts.shape
        R_granger = np.zeros((n, m, m))

        for l in range(n):
            for i in range(m):
                for j in range(m):
                    ts_nij = np.dstack((self.ts[l,:,i], self.ts[l,:,j]))[0]
                    results = tsa_tools.grangercausalitytests(ts_nij, maxlag=max_lag, verbose=0)
                    pvalue_max = max([result[key][0]["ssr_ftest"][1] for key in result])
                    R_granger[l,i,j] = pvalue_max
        
        return R_granger