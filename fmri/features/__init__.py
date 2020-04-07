import numpy as np
from scipy.linalg import fractional_matrix_power as powm

import networkx as nx

import fmri.features.spectral
import fmri.features.differential
import fmri.features.graph
import fmri.features.topological

class FeatureExtractor(
    spectral.SpectralFeatureExtractor,
    differential.DifferentialFeatureExtractor,
    graph.GraphFeatureExtractor,
    topological.TopologicalFeatureExtractor
    ):

    def __init__(self, A, cache=False):
        self.A = A
        self.cache = cache
