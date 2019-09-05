import numpy as np

from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

import fmri.connectivity.correlation

class CorrelationConnectivity():

    def correlation(self, estimator="maximum_likelihood", assume_centered=False):

        if estimator=="corrcoef":
            A = np.corrcoef(self.ts.T)
        elif estimator=="maximum_likelihood":
            correlation_measure = ConnectivityMeasure(kind="correlation", cov_estimator=EmpiricalCovariance(assume_centered=assume_centered))
            A = correlation_measure.fit_transform([self.ts])[0]
        elif estimator=="ledoit_wolf":
            correlation_measure = ConnectivityMeasure(kind="correlation", cov_estimator=LedoitWolf(assume_centered=assume_centered))
            A = correlation_measure.fit_transform([self.ts])[0]
        else:
            raise ValueError("Estimator should be 'corrcoef', 'maximum_likelihood', or 'ledoit_wolf'")

        return A