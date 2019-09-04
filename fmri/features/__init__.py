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

    def __init__(self, G, cache=False):
        self.G = G
        self.cache = cache
