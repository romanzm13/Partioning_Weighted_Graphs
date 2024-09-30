# Partitioning Weighted Graphs
1. ProposedAlgorithms.py file contains the implementation of proposed algorithms for community detection in weighted networks. 
Among these algorithms is the generalization of Newman's optimal modularity algorithm for the case of weighted graphs (named GeNA).
In addition, there is the implementation of an algorithm to decrease the number of communities of any partition, called FUSE,
through sums of inter-community modularity. In contrast, there is an algorithm to increase the number of communities of any partition, named ADD,
using a proposed dissimilarity measure based on the magnitud of the components of modularity matrix. Using these three algorithms we present the
method ReVAM (Resolution Variation Algorithm by Modularity) to obtain straightforward a partition of the weigthed grapn in a wanted number of communities.
Finally, proposed functions can be found to calculate the level of membership of each node with respect to its community.
2. Regarding ComparisonResolutionAlgorithms.py, this shows the performance of ReVAM compared with Louvain, Combo, Spectral Clustering and Leiden algorithms.
3. In AlgorithmsApplication.py file is presented a case of application of the algorithms and functions previously stated. This application
consists on generating partitions of a network from state of Guanajuato, constructed through geographic adjacencies of
municipalities and weighted using incidence data of COVID-19.
