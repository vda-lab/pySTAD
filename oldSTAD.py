import pandas as pd
import numpy as np
import igraph as ig
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import scipy
from copy import deepcopy
import pprint

history_x = []
history_y = []
history_graph = []

debug = False

class MyTakeStep(object):
    def __init__(self, stepsize=1000, min_links=0, max_links=99999):
        self.stepsize = stepsize
        self.min_links = min_links
        self.max_links = max_links
    def __call__(self, x):
        s = self.stepsize
        x = int(np.random.normal(x, s))
        if ( x < self.min_links ):
            x = self.min_links
        if ( x > self.max_links ):
            x = self.max_links
        return x

def normalize(x, domain_min, domain_max, range_min, range_max):
    '''normalize
    Method for changing values from domain to range (like P5 map)
    '''
    return (range_max-range_min)/(domain_max-domain_min)*(x-domain_min)+range_min

def normalize_matrix(matrix):
    max_value = np.max(matrix)
    return matrix/max_value

def get_bin(x, bin_size):
    return x/bin_size

def assign_bins(data, nr_bins):
    bins = list(pd.cut(data['lens'], nr_bins, labels=range(nr_bins)))
    data['bin'] = bins
    return data

def create_dist_matrix(values):
    return euclidean_distances(values)

def create_network(x, mst, non_mst):
    '''create_network
    Generate a network with a given number of non-mst links
    '''
    extras = non_mst.take(range(0,x), axis=0)

    network = np.vstack((mst, extras))

    links = network[:,[0,1]].astype(int)
    nodes = np.unique(np.hstack((links[:,0], links[:,1])))
    link_tuples = list(map(tuple, links))
    network_graph = ig.Graph()
    network_graph.add_vertices(nodes)
    network_graph.add_edges(link_tuples)

    similarities = network[:,2]
    distances = np.max(similarities) - similarities
    network_graph.es["distance"] = distances
    network_graph.es["similarity"] = similarities
    network_graph.es['color'] = np.repeat('black',len(distances))

    # For debugging => remove afterwards
    for e in network_graph.es:
        if len(non_mst[(non_mst[:,0] == e.tuple[0])
                        & (non_mst[:,1] == e.tuple[1])]) == 1:
            e['color'] = 'red'
        else:
            e['color'] = 'blue'

    return network_graph

def dist2sim(dist_matrix):
    max_value = np.max(dist_matrix)
    return max_value-dist_matrix

def alter_dist_matrix_1step(dist_matrix, data):
    '''create_dist_matrix_for_bins
    Takes the full distance matrix, and:
    * leaves distances between points within the same bin and adjacent bins alone
    * sets distances between points in _nonadjacent_ bins to NA
    '''
    dist_matrix_for_bins = deepcopy(dist_matrix)
    max_idx = len(dist_matrix)

    # Should max_value be the maximal distance within the bins, or overall?
    max_value = np.max(dist_matrix)
    # max_value = 0
    # bins = range(0, np.max(data['bin']+1))
    # for b in bins:
    #     elements = list(data.loc[data['bin'] == b]['id'])
    #     for i in range(len(elements)):
    #         for j in range(i+1,len(elements)):
    #             if dist_matrix[elements[i]][elements[j]] > max_value:
    #                 max_value = dist_matrix[elements[i]][elements[j]]
    #### end max_value

    for i in range(max_idx):
        record_i = data.iloc[[i]]
        for j in range(i+1, max_idx):
            record_j = data.iloc[[j]]
            dist = abs(int(record_i['bin']) - int(record_j['bin']))
            if dist > 1:
                dist_matrix_for_bins[i][j] = 100*max_value
                dist_matrix_for_bins[j][i] = 100*max_value
    return dist_matrix_for_bins

def create_mst(dist_matrix):

    pd_dist_matrix = pd.DataFrame(dist_matrix)
    # Transform distance matrix
    # input: nxn distance matrix
    # output: numpy array
    # columns:   from   to   weight    mst   order
    #   22     118  0.817978  0   0
    #   14     118  0.831283  0   1
    #   22     122  0.832234  0   2
    #   33     118  0.836029  0   3
    #   22     108  0.836048  0   4
    # 'mst' all 0, but will become 1 for all links in the MST
    # 'order' = order in which to add links (because df is sorted by value)
    pd_melted_dist_matrix = pd_dist_matrix.stack()
    pd_melted_dist_matrix = pd_melted_dist_matrix.reset_index()
    mdm = pd_melted_dist_matrix.values # mdm = melted_dist_matrix

    # Only keep those distances where from_node < to_node (symmetrical distance matrix)
    cmdm = mdm[mdm[:,0] < mdm[:,1]]

    # Add column of zeros, indicating that none of the links are part of the MST
    # (will be corrected further down)
    z = np.zeros((len(cmdm),4))
    z[:,:-1] = cmdm
    cmdm = z

    # Add column representing the order in which links are added in addition to the MST
    # Depends on the weight
    cmdm = cmdm[cmdm[:,2].argsort()[::-1]] # order by weight
    # cmdm = cmdm[cmdm[:,2].argsort()[::]] # order by weight
    a = np.array(range(len(cmdm))).reshape(len(cmdm),1)
    cmdm = np.hstack((cmdm, a))

    # Order by from, then by to (columns 0 and 1), so that we can easily compare with the unit-graph
    # see http://numpy-discussion.10968.n7.nabble.com/how-to-do-a-proper-2-column-sort-on-a-2-dimensional-array-td19927.html
    i = np.lexsort((cmdm[:,1], cmdm[:,0]))
    cmdm = cmdm[i]

    # Mark the MST links
    # e.g. link 33->118 is in MST
    ## columns:   from   to   weight    mst   order
    #   22     118  0.817978  0   0
    #   14     118  0.831283  1   1
    #   22     122  0.832234  1   2
    #   33     118  0.836029  0   3
    #   22     108  0.836048  1   4
    if ( debug ):
        print("Working on the network...")
    links = cmdm[:,[0,1,2]]

    edges = links[:,[0,1]].astype(int)
    vertices = np.unique(np.hstack((edges[:,0], edges[:,1])))
    edge_tuples = list(map(tuple, edges))
    full_graph = ig.Graph()
    full_graph.add_vertices(vertices)
    full_graph.add_edges(edge_tuples)

    distances = links[:,2]
    similarities = np.max(distances) - distances
    full_graph.es["distance"] = distances
    full_graph.es["similarity"] = similarities

    mst_graph = full_graph.spanning_tree(weights = distances)

    if ( debug ):
        print("Updating cmdm with mst information")
        print("Number of edits to make: ", str(len(mst_graph.es)))
    for e in mst_graph.es:
        record_to_change = np.where((cmdm[:,0] == e.tuple[0]) & (cmdm[:,1] == e.tuple[1]))[0][0]
        cmdm[record_to_change][3] = 1

    non_mst = cmdm[np.where(cmdm[:,3] == 0)]
    non_mst = non_mst[non_mst[:,2].argsort()]
    mst = cmdm[np.where(cmdm[:,3] == 1)]

    # dm_distances array = distance matrix distances = what to compare each proposal to
    dm_distances = cmdm[:,2]

    return [mst_graph, mst, non_mst, cmdm, dm_distances]

# def remove_mst_links_between_communities(mst_graph):
#     a = mst_graph.community_walktrap(weights='distance',steps=20)

# def create_dist_matrix_for_connected_components(dist_matrix, data):
#     return false

def objective_function(x, args):
    '''objective_function
    Function to be minimized: distance between original distance matrix and graph-based distance matrix
    '''
    global history_x
    global history_y
    global history_graph

#     global mst
#     global non_mst
#     global cmdm

    global debug

    mst=args['mst']
    non_mst=args['non_mst']
    cmdm=args['cmdm']
    dm_distances=args['dm_distances']

    i = int(x)
    if ( debug ):
        print("Running objective function for: ", str(i))

    proposal_graph = create_network(i, mst, non_mst)

    proposal_dist = proposal_graph.shortest_paths()
    l = len(proposal_dist)
    graph_distances = []
    for from_node in range(l):
        for to_node in range(from_node + 1, l):
            graph_distances.append([from_node, to_node, proposal_dist[from_node][to_node]])

    graph_distances = np.asarray(graph_distances)

    # Order by 'from' and 'to'
    s = np.lexsort((graph_distances[:,1], graph_distances[:,0]))
    graph_distances = graph_distances[s]

    corr = abs(np.corrcoef(graph_distances[:,2], dm_distances)[0][1])
    if ( debug ):
        print("nr edges = ", str(i), " -> correlation = ", str(corr))

    history_graph.append(list(map(lambda e:e.tuple, proposal_graph.es)))
    history_x.append(i)
    history_y.append(corr)
    return (1-corr) # Because basinhopping MINIMIZES

def create_complete_plot(dist_matrix, res = 50):
    '''create_complete_plot
    Creates overview picture of effect of nr of non-mst links on objective function
    '''
    global history_x
    global history_y
    global history_graph

    global debug

    history_x = []
    history_y = []
    history_graph = []


    resolution = res # How many points do we want to compute?
    mst_graph, mst, non_mst, cmdm, dm_distances = create_mst(dist_matrix)

    search_space = list(range(len(cmdm) - len(mst)))
    stepsize = int((search_space[-1]-search_space[0])/resolution)

    if ( debug ):
        print(int(search_space[0]))
        print(int(search_space[-1]))
        print(stepsize)

    args = {'mst':mst, 'non_mst':non_mst, 'cmdm':cmdm, 'dm_distances':dm_distances}
    for i in range(0,stepsize*resolution,stepsize):
        i = str(i)
#         if ( debug ):
#             print("Running objective function for: ", str(i))
        objective_function(i, args)

    return [history_x, history_y, history_graph]

def find_stad_optimum(dist_matrix, T=0.5, interval=100):
    '''find_stad_optimum
    Finds the optimum nr of links
    '''
    global history_x
    global history_y
    global history_graph
    history_x = []
    history_y = []
    history_graph = []

    global debug

    mst_graph, mst, non_mst, cmdm, dm_distances = create_mst(dist_matrix)

    search_space = list(range(len(cmdm) - len(mst)))
    stepsize = int(search_space[-1]-search_space[0])/100

    print("Stepsize: ", str(stepsize))
    print("Search space min: ", str(search_space[0]))
    print("Search space max: ", str(search_space[-1]))

    print("history_y: ", history_y)

    minimizer_kwargs = {'args':{'mst':mst, 'non_mst':non_mst, 'cmdm':cmdm, 'dm_distances':dm_distances}}
    result = scipy.optimize.basinhopping(
                    objective_function, 0,
                    minimizer_kwargs=minimizer_kwargs, #disp=True,
                    T=T, interval=interval,
                    take_step=MyTakeStep(stepsize=stepsize, min_links=search_space[0], max_links=search_space[-1]))
    print([int(result.x[0]), result.fun])
    return [int(result.x[0]), result.fun]


# def alter_dist_matrix_phase1(dist_matrix, data):
#     '''create_dist_matrix_for_bins
#     Takes the full distance matrix, and:
#     * leaves distances between points within the same bin alone
#     * increases distances between points in _adjacent_ bins with max value
#     * sets distances between points in _nonadjacent_ bins to NA
#     '''
#     dist_matrix_for_bins = deepcopy(dist_matrix)
#     max_idx = len(dist_matrix)

#     # Should max_value be the maximal distance within the bins, or overall?
#     max_value = np.max(dist_matrix)
#     # max_value = 0
#     # bins = range(0, np.max(data['bin']+1))
#     # for b in bins:
#     #     elements = list(data.loc[data['bin'] == b]['id'])
#     #     for i in range(len(elements)):
#     #         for j in range(i+1,len(elements)):
#     #             if dist_matrix[elements[i]][elements[j]] > max_value:
#     #                 max_value = dist_matrix[elements[i]][elements[j]]
#     #### end max_value

#     for i in range(max_idx):
#         record_i = data.iloc[[i]]
#         for j in range(i+1, max_idx):
#             record_j = data.iloc[[j]]
#             dist = abs(int(record_i['bin']) - int(record_j['bin']))
#             if dist == 1:
#                 dist_matrix_for_bins[i][j] += 2*max_value
#                 dist_matrix_for_bins[j][i] += 2*max_value
#             elif dist > 1:
#                 dist_matrix_for_bins[i][j] = None
#                 dist_matrix_for_bins[j][i] = None
#     return dist_matrix_for_bins

# def add_communities_to_data(communities, data):
#     data['community'] = [None] * len(data)
#     for i, comm in enumerate(communities):
#         for node in comm:
#             data['community'][node] = i
#     return data

# def split_connected_components(graph, bins, dist_matrix, communities, data):
#     #### Remove bad links
#     # Make communities a hash so that I can search quicker
#     community2node_hash = {}
#     node2community_hash = {}
#     links_to_remove = []

#     for i,comm in enumerate(communities):
#         for node in comm:
#             node2community_hash[node] = i
#             community2node_hash[i] = node
#     for e in graph.es:
#         if not node2community_hash[e.tuple[0]] == node2community_hash[e.tuple[1]]:
#             links_to_remove.append((e.tuple[0], e.tuple[1]))
#     graph.delete_edges(links_to_remove)
#     return graph

# def connect_connected_components(graph, bins, dist_matrix, communities, data):
#     #### Remove bad links
#     # Make communities a hash so that I can search quicker
#     community2node_hash = {}
#     node2community_hash = {}
#     #### Link the connected components together again
#     # To make searching easier, check which bin each connected comp is in
#     bin2communities_hash = {}
#     community2bin_hash = {}

#     for b in bins:
#         bin2communities_hash[b] = []
#     for i, comm in enumerate(communities):
#         b = data['bin'][comm[0]]
#         bin2communities_hash[b].append(i)
#         community2bin_hash[i] = b
#     # print("community2bin_hash")
#     # pprint.pprint(community2bin_hash)
#     # print('---')
#     # print("bin2communities_hash:")
#     # pprint.pprint(bin2communities_hash)

#     # For each community
#     links_to_add = []
#     for i, comm in enumerate(communities):
#         # print('----------------')

#         # which are the communities in the same and the adjacent bins
#         # print("DEBUG: looking at community ", i)
#         communities_to_search = []

#         b = community2bin_hash[i]
#         # print("   in bin:", b)
#         communities_to_search = deepcopy(bin2communities_hash[b])
#         # print("   communities initial:  ")
#         # print(sorted(set(communities_to_search)))

#         if b > 0:
#             # print("  bin-1:", b-1)
#             communities_to_search += bin2communities_hash[b-1]
#             # print(sorted(set(bin2communities_hash[b-1])))
#         if b < max(bins):
#             # print("  bin+1:", b+1)
#             communities_to_search += bin2communities_hash[b+1]
#             # print(sorted(set(bin2communities_hash[b+1])))
#         communities_to_search = set(communities_to_search)
#         nodes_to_search = []
#         for c in communities_to_search:
#             for n in communities[c]:
#                 nodes_to_search.append(n)
#         # print(nodes_to_search)
#         # print(len(nodes_to_search))
#         # remove nodes from the comm we're looking at itself
#         nodes_to_search = [x for x in nodes_to_search if x not in comm]
#         # print("nr nodes_to_search: ", len(nodes_to_search))
#         # print("DEBUG: new nodes to search: ")
#         # print(nodes_to_search)
#         # print(len(nodes_to_search))

#         if not len(nodes_to_search) == 0:
#             min_distance = 999999
#             node_with_link = None
#             for node in comm:
#                 distances = list(map(lambda x:dist_matrix[node][x], nodes_to_search))
#                 min_distance_for_node = np.min(distances)
#                 if min_distance_for_node < min_distance:
#                     min_distance_idx = np.argmin(distances)
#                     min_distance = min_distance_for_node
#                     node_with_link = node
#             links_to_add.append((node_with_link, nodes_to_search[min_distance_idx]))
#     # print("DEBUG: links to add")
#     # print(links_to_add)
#     graph.add_edges(links_to_add)
#     return graph

# def alter_dist_matrix_phase2(dist_matrix, communities):
#     '''
#     alter_dist_matrix_phase2
#     * takes original (!) distance matrix
#     * adds max_value to all distances between points that are not in the same community
#     '''
#     dist_matrix_for_communities = deepcopy(dist_matrix)
#     max_idx = len(dist_matrix)
#     print(max_idx)
#     max_value = np.max(dist_matrix[~np.isnan(dist_matrix)]) # need to filter out the Nan...
#     print(max_value)
#     for i in range(max_idx):
#         for j in range(i+1, max_idx):
#             found_in_same_community = False
#             for comm in communities:
#                 if not found_in_same_community:
# #                     if all(elem in comm for elem in [i,j]):  # check this!!!
#                     if set([i,j]) <= set(comm):
#                         found_in_same_community = True

#             if not found_in_same_community:
#                 dist_matrix_for_communities[i][j] += 2*max_value
#                 dist_matrix_for_communities[j][i] += 2*max_value
#     return dist_matrix_for_communities

# def alter_dist_matrix_phase2(dist_matrix, communities):
#     dist_matrix_for_communities = deepcopy(dist_matrix)
#     max_idx = len(dist_matrix)
#     for i in range(max_idx):
#         for j in range(i+1, max_idx):
#             found_in_same_community = False
#             for comm in communities:
#                 if not found_in_same_community:
#                     if all(elem in comm for elem in [i,j]):  # check this!!!
#                         found_in_same_community = True

#             if not found_in_same_community:
#                 dist_matrix_for_communities[i][j] *= 10
#                 dist_matrix_for_communities[j][i] *= 10
#     return dist_matrix_for_communities
