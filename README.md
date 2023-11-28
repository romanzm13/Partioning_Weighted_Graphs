#Partioning_Weighted_Graphs
1. The 'ProposedAlgorithms.py' file contains the implementation of proposed algorithms for community detection in weighted networks. 
Among these algorithms is the generalization of Newman's optimal modularity algorithm for the case of weighted graphs (named GAWG).
On the other hand, there is the implementation of an algorithm to decrease the number of communities of any partition (called CUAM),
through sums of inter-community modularity. In contrast, there is an algorithm to increase the number of communities of any partition
(named CDAD), using a proposed dissimilarity measure that based on componentes of modularity matrix.
Finally, proposed functions can be found to calculate the belonging level of each node with respect to its community.
2. In 'AlgorithmsApplication.py' file is presented a case of application of the algorithms and functions previously stated. This application
consists on generating partitions of a network from state of Guanajuato, constructed through geographic adjacencies of
municipalities and weighted using incidence data of COVID-19.
