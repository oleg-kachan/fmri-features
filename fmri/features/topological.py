import numpy as np
from scipy.linalg import fractional_matrix_power as powm
from scipy.stats import skew as skewness, kurtosis

import networkx as nx

from ripser import ripser
from persim import PersImage

quantities = {
    "persistence": lambda x: x[:,1] - x[:,0],
    "midlife": lambda x: (x[:,0] + x[:,1]) / 2,
    "multlife": lambda x: x[:,1] / x[:,0]
}

stats = {
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "mean": np.mean,
    "std": np.std,
    "skewness": skewness,
    "kurtosis": kurtosis,
    "median": np.median,
    "entropy": lambda x: -(x @ log(x)),
    "count": lambda x: x.shape[0]
}

functions = {
    "linear": lambda x, **kwargs: linear(x, **kwargs),
    "pearson": lambda x, **kwargs: pearson(x, **kwargs)
}

def linear(X, a=0, b=0):
    return X * a + b

def pearson(X, n=150, z_alpha=1.96):
    def z(x):
        return np.arctanh(x)

    def r(x):
        return np.tanh(x)

    def error(x):
        return (r(z(x) + z_alpha/np.sqrt(n - 3)) - r(z(x) - z_alpha/np.sqrt(n - 3))) / 2

    return error(X)

def log(x): 
    return np.log(x + 1e-100)

def get_thresholds(m=99):
    threshold = []
    for i in range(m):
        threshold.append((i + 1) / (m + 1))

    return threshold

def persistence(diagram, curve, quantity, thresholds):

    if thresholds==None:
        critical_points = np.unique(diagram.reshape(-1))
    else:
        critical_points = thresholds

    x = {}
    for t in critical_points:
        x[t] = 0

    for t in critical_points:
        
        for interval in diagram:
            start, end = interval[0], interval[1]
            
            # persistence curves
            if curve=="persistence":

                if quantity=="betti":
                    x_inc = 1
                else:
                    x_inc = quantities[quantity](interval.reshape(1,-1)).reshape(-1)[0]
                
            # entropy curves
            elif curve=="entropy":

                p = quantities[quantity](interval.reshape(1,-1)).reshape(-1)[0]
                p_norm = quantities[quantity](diagram).sum()
                    
                p = p / p_norm
                x_inc = -(p * log(p))
            
            if ((start <= t) & (t < end)):
                x[t] = x[t] + x_inc
                
    keys, values = zip(*x.items())

    return np.array(values)


class TopologicalFeatureExtractor():

    def get_diagram(self, k_max=2, inf=1.0):
        diagram = ripser(self.A[0], maxdim=k_max, thresh=1.0, distance_matrix=True)["dgms"]
        diagram[0][::-1][0][1] = inf # remove inf

        return diagram

    def betti_curve(self, betti_number=0):

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

        betti_curve = []

        for A_t in self.A:
            G_t = nx.from_numpy_matrix(A_t)
            betti_curve.append(betti(G_t))
        
        return np.array(betti_curve)

    
    def statistics(self, diagram, k_max=None, quantity="persistence", statistics=["mean", "sum"], f="linear", **kwargs):

        if k_max is None:
            k_max = len(diagram) - 1

        # init statistics matrix
        matrix_statistics = np.zeros((k_max+1, len(statistics)))
        
        for k in range(k_max+1):

            # filter a diagram according to a confidence function
            p = quantities["persistence"](diagram[k])
            diagram_k = diagram[k][p > functions[f](p, **kwargs)]

            # compute a quantity out of a diagram
            q = quantities[quantity](diagram_k)

            # compute the quantity's statistics
            for j, statistic in enumerate(statistics):
                matrix_statistics[k,j] = stats[statistic](q)

        # replace nans
        matrix_statistics = np.nan_to_num(matrix_statistics)

        return matrix_statistics

    
    def euler_curve(self, diagram, m=99, k_max=None, f="linear", **kwargs):

        if k_max is None:
            k_max = len(diagram) - 1

        thresholds = get_thresholds(m)
        
        # init output matrix
        betti_curve = np.empty((0, len(thresholds)))

        for k in range(k_max+1):

            # filter a diagram according to a function
            p = quantities["persistence"](diagram[k])
            diagram_k = diagram[k][p > functions[f](p, **kwargs)]

            betti_curve_k = persistence(diagram_k[::-1], "persistence", "betti", thresholds)
            betti_curve = np.vstack((betti_curve, betti_curve_k))

        euler_curve = betti_curve.sum(axis=0)

        return euler_curve


    def persistence_curve(self, diagram, quantity="betti", m=99, k_max=None, f="linear", **kwargs):

        if k_max is None:
            k_max = len(diagram) - 1

        thresholds = get_thresholds(m)
        
        # init output matrix
        persistence_curve = np.empty((0, len(thresholds)))
        
        for k in range(k_max+1):

            # filter diagram according to function
            p = quantities["persistence"](diagram[k])
            diagram_k = diagram[k][p > functions[f](p, **kwargs)]

            persistence_curve_k = persistence(diagram_k[::-1], "persistence", quantity, thresholds)
            persistence_curve = np.vstack((persistence_curve, persistence_curve_k))

        return persistence_curve


    def entropy_curve(self, diagram, quantity="persistence", m=99, k_max=None, f="linear", **kwargs):

        if k_max is None:
            k_max = len(diagram) - 1

        thresholds = get_thresholds(m)

        # init output matrix
        entropy_curve = np.empty((0, len(thresholds)))

        for k in range(k_max+1):

            # filter diagram according to function
            p = quantities["persistence"](diagram[k])
            diagram_k = diagram[k][p > functions[f](p, **kwargs)]

            entropy_curve_k = persistence(diagram_k[::-1], "entropy", quantity, thresholds)
            entropy_curve = np.vstack((entropy_curve, entropy_curve_k))

        return entropy_curve

    
    def vector(self, diagram, quantity="persistence", aggregation="mean", thresholds=None, k_max=2, stride=1.5, axis=0, f="linear", **kwargs):

        def bins(diagram, quantity, aggregation, thresholds, stride, axis):

            def set_bins(diagram, quantity, thresholds, stride, axis):
                
                # set bin width w.r.t stride
                bin_width = thresholds[1] - thresholds[0]
                bin_width_strided = bin_width * stride
                bin_pad = (bin_width_strided - bin_width) / 2

                # add point (b, d) to a bin if lifetime is in interval
                bins = []
                for i in range(thresholds.shape[0]-1):
                    start = thresholds[i] - bin_pad
                    end = thresholds[i+1] + bin_pad
                    
                    if axis==0:
                        bins.append(diagram[(diagram[:,0] >= start) & (diagram[:,0] <= end)])
                    
                    elif axis==1:
                        q = quantities[quantity](diagram)
                        bins.append(diagram[(q >= start) & (q <= end)])

                return bins
        
            # bin persistence diagram
            pd_binned = set_bins(diagram, quantity, thresholds, stride, axis)

            pers_vec = np.zeros(thresholds.shape[0]-1)

            for i, arr_bin in enumerate(pd_binned):
                x = quantities[quantity](arr_bin)
                pers_vec[i] = stats[aggregation](x)

            pers_vec = np.nan_to_num(pers_vec)
            
            #if normalized:
            #    pers_vec = pers_vec / pers_vec.sum()
            
            return pers_vec

        # init output matrix
        matrix_bins = np.empty((0, thresholds.shape[0]-1))
        
        for k in range(k_max+1):

            # filter diagram according to function
            p = quantities["persistence"](diagram[k])
            diagram_k = diagram[k][p > functions[f](p, **kwargs)]

            matrix_bins_k = bins(diagram_k, quantity, aggregation, thresholds, stride, axis)
            matrix_bins = np.vstack((matrix_bins, matrix_bins_k))

        return matrix_bins
    

    def persistence(self, k_max=1, type="standard", representation="surface", shape=(16,16)):

        if representation=="surface":
            pi = PersImage(spread=0.025, pixels=shape, verbose=False)
            result = np.array(pi.transform(diagrams[1]))
        elif representation=="landscape":
            raise NotImplementedError() # TODO
        elif representation=="diagram":
            result = diagrams
        else:
            raise ValueError("Representation should be 'vector', 'surface', 'landscape', or 'diagram'")
    
        return result