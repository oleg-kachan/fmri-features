import numpy as np

import fmri.connectivity.correlation

class CorrelationConnectivity():

    def correlation(self, estimator="maximum_likelihood"):

        if estimator=="maximum_likelihood":
            A = np.corrcoef(self.ts)
        elif estimator=="ledoit_wolf":
            raise NotImplementedError # TODO
        else:
            raise ValueError("Estimator should be 'maximum_likelihood' or 'ledoit_wolf'")

        return A