import numpy as np
import networkx as nx
import pickle

import fmri.connectivity
import fmri.features

A = np.array([
    [0, 0.9, 0.7, 0],
    [0.9, 0, 0, 0],
    [0.7, 0, 0, 0.4],
    [0, 0, 0.4, 0],
])

# read time series from disk
time_series = pickle.load(open("_data/sub-T001.pkl", "rb"))
print("TS", time_series.shape)

# compute connectivity matrix out of time series
Connectivity = fmri.connectivity.Connectivity(time_series.T)
A = Connectivity.correlation(estimator="maximum_likelihood")
A = Connectivity.threshold(A, threshold=0.7)
#print("A", A.shape)

# set networkx Graph object from connectivity matrix
G = nx.from_numpy_array(A)
# print("G", G.number_of_edges())

# initialize Extractor object networkx Graph object 
Extractor = fmri.features.FeatureExtractor(G)

print("SPECTRAL\r\n---")
print("Spectre of adjacency matrix", Extractor.spectral_adjacency())
print("Spectre of Laplacian", Extractor.spectral_laplacian())
print("Spectre of normalized Laplacian", Extractor.spectral_laplacian_normalized())
print("Spectre of non-backtracking cycles matrix", Extractor.spectral_non_backtracking())

print("\r\nDIFFERENTIAL\r\n---")
print("Heat kernel signature", Extractor.hks(t=5))
print("Wave kernel signature", Extractor.wks(t=0.5))

print("\r\nGRAPH-THEORETIC\r\n---")
print("Degree", Extractor.degree())
print("Betweenness centrality", Extractor.betweenness_centrality())
print("Closeness centrality", Extractor.closeness_centrality()) # do not support weights
print("Clustering coefficient", Extractor.clustering_coefficient())
print("Neighbor degree", Extractor.neighbor_degree())
#print("Average shortest path", Extractor.average_shortest_path())
print("Global efficiency", Extractor.global_efficiency())  # do not support weights
print("---")
print("Eigenvector centrality", Extractor.eigenvector_centrality())
#print("Second-order centrality", Extractor.second_order_centrality()) # do not support weights
print("---")
#print("Ollivier-Ricci curvature", Extractor.ollivier_ricci_curvature())
#print("Forman-Ricci curvature", Extractor.forman_ricci_curvature())

print("\r\nTOPOLOGICAL\r\n---")
print("Betti-0 curve", Extractor.betti_curve(betti_number=0, shape=18))
print("Betti-1 curve", Extractor.betti_curve(betti_number=1, shape=14))
print("Persistent surface of PD_0 + PD_1", Extractor.persistence(k=[0,1], type="standard", representation="surface", shape=(10,10)))
print("Persistent landscape of PD_2", Extractor.persistence(k=[2], type="standard", representation="landscape", shape=(50)))
print("Persistent surface of EPD_1", Extractor.persistence(k=[1], type="extended", representation="surface", shape=(20,20)))
print("Persistent landscape of EPD_2", Extractor.persistence(k=[2], type="extended", representation="landscape", shape=(25)))