import numpy as np

from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

class SymmetricConnectivity():

    def correlation(self, estimator="maximum_likelihood", assume_centered=False):

        if estimator=="maximum_likelihood":
            correlation_measure = ConnectivityMeasure(kind="correlation", cov_estimator=EmpiricalCovariance(assume_centered=assume_centered))
        elif estimator=="ledoit_wolf":
            correlation_measure = ConnectivityMeasure(kind="correlation", cov_estimator=LedoitWolf(assume_centered=assume_centered))
        else:
            raise ValueError("Estimator should be 'maximum_likelihood' or 'ledoit_wolf'")

        R = np.nan_to_num(correlation_measure.fit_transform(self.ts))

        return R