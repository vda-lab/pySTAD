import pandas as pd
import igraph as ig
import numpy as np
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy import optimize

import panel as pn
import vega
pn.extension('vega')

########
#### Auxiliary functions
########
def calculate_highD_dist_matrix(data):
    non_normalized = euclidean_distances(data)
    max_value = np.max(non_normalized)
    return non_normalized/max_value

def rgb_to_hsv(rgb):
    r = float(rgb[0])
    g = float(rgb[1])
    b = float(rgb[2])
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, v = high, high, high

    d = high - low
    s = 0 if high == 0 else d/high

    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, v

def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return list(int(hex[i:i+int(hlen/3)], 16) for i in range(0, hlen, int(hlen/3)))

def hex_to_hsv(hex):
    return rgb_to_hsv(hex_to_rgb(hex))

def normalise_number_between_0_and_1(nr, domain_min, domain_max):
    return (nr-domain_min)/(domain_max - domain_min)

def normalise_number_between_0_and_255(nr, domain_min, domain_max):
    return 255*normalise_number_between_0_and_1(nr, domain_min, domain_max)

########
#### Load test data
########
def load_testdata(dataset):
    if dataset == 'horse':
        data = pd.read_csv('data/horse.csv', header=0)
        data = data.sample(n=1000)
        values = data[['x','y','z']].values.tolist()
        x_min = min(data['x'])
        x_max = max(data['x'])
        # zs = data['z'].values
        lens = data['x'].map(lambda x:normalise_number_between_0_and_255(x, x_min, x_max)).values
        return(values, lens)
    elif dataset == 'simulated':
        data = pd.read_csv('data/sim.csv', header=0)
        values = data[['x','y']]
        lens = np.zeros(1)
        return(values, lens)
    elif dataset == 'circles':
        data = pd.read_csv('data/five_circles.csv', header=0)
        values = data[['x','y']].values.tolist()
        lens = data['hue'].map(lambda x:hex_to_hsv(x)[0]).values
        return(values, lens)
    else:
        print("Dataset not known")

########
#### Create MST
########
def matrix_to_topright_array(matrix):
    for i, vector in enumerate(matrix):
        for j, value in enumerate(vector):
            if ( j > i ):
                yield value

def create_mst(dist_matrix, lens = [], features = {}):
    complete_graph = ig.Graph.Full(len(dist_matrix[0]))
    if len(lens) > 0:
        complete_graph.vs["lens"] = lens
    if not features == {}:
        for f in list(features.keys()):
            complete_graph.vs[f] = features[f]
    complete_graph.es["distance"] = list(matrix_to_topright_array(dist_matrix))
    return complete_graph.spanning_tree(weights = list(matrix_to_topright_array(dist_matrix)))

########
#### Alter distance matrix to incorporate lens
########
# There are 2 options to incorporate a lens:
# 1. Set all values in the distance matrix of datapoints that are in non-adjacent
#    bins to 1000, but leave distances within a bin and between adjacent
#    bins untouched.
# 2. a. Set all values in the distance matrix of datapoints that are in non-adjacent
#       bins to 1000, add 2 to the values of datapoints in adjacent bins, and
#       leave distances of points in a bin untouched.
#    b. Build the MST
#    c. Run community detection
#    d. In the distance matrix: add a 2 to some of the data-pairs, i.e. to those
#       that are in different communities.
#    e. Run the regular MST again on this new distance matrix (so is the same
#       as in step a, but some of the points _within_ a bin are also + 2)
def assign_bins(lens, nr_bins):
    # np.linspace calculates bin boundaries
    # e.g. bins = np.linspace(0, 1, 10)
    #  => array([0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
    #            0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ])
    bins = np.linspace(min(lens), max(lens), nr_bins)

    # np.digitize identifies the correct bin for each datapoint
    # e.g.
    #  => array([3, 9, 7, 8, 8, 3, 9, 6, 4, 3])
    return np.digitize(lens, bins)

def create_lensed_distmatrix_1step(matrix, assigned_bins):
    '''
    This will set all distances in non-adjacent bins to 1000. Data was
    normalised between 0 and 1, so 1000 is far (using infinity gives issues in
    later computations).
    Everything after this (building the MST, getting the list of links, etc)
    will be based on this new distance matrix.
    '''
    size = len(matrix)
    single_step_addition_matrix = np.full((size,size), 1000)

    for i in range(0, size):
        for j in range(i+1,size):
            if ( abs(assigned_bins[i] - assigned_bins[j]) <= 1 ):
                single_step_addition_matrix[i][j] = 0
    return matrix + single_step_addition_matrix

########
#### Draw the graph
########
def create_vega_nodes(graph):
    output = []
    # The following will automatically pick up the lens...
    feature_labels = graph.vs()[0].attributes().keys()
    for v in graph.vs():
        node = {"name": v.index}
        for f in feature_labels:
            node[f] = v.attributes()[f]
        output.append(node)
    return output

def create_vega_links(graph):
    output = []
    for e in graph.es():
        output.append({"source": e.source, "target": e.target, "value": e.attributes()['distance']})
    return output

def create_gephi_files(graph, filename):
    with open(filename + '_nodes.csv', 'w') as f:
        if 'lens' in graph.vs()[0].attributes():
            f.write("id\tlens\n")
            counter = 0
            for v in graph.vs():
                f.write(str(counter) + "\t" + str(v.attributes()['lens']) + "\n")
                counter += 1
        else:
            f.write("id\n")
            counter = 0
            for v in graph.vs():
                f.write(str(counter) + "\n")
                counter += 1

    with open(filename + '_edges.csv', 'w') as f:
        f.write("source\ttarget\tvalue\n")
        for e in graph.es():
            f.write(str(e.source) + "\t" + str(e.target) + "\t" + str(e.attributes()['distance']) + "\n")

def draw_stad(graph):
    strength_picker = pn.widgets.IntSlider(name='Attraction strength', start=-10, end=1, step=1, value=-3)
    distance_picker = pn.widgets.IntSlider(name='Distance between nodes', start=1, end=30, step=1, value=15)
    radius_picker = pn.widgets.IntSlider(name='Node radius', start=1, end=5, step=1, value=5)
    theta_picker = pn.widgets.FloatSlider(name='Theta', start=0.1, end=1.5, step=0.1, value=0.9)
    distance_max_picker = pn.widgets.IntSlider(name='Max distance cap', start=0, end=300, step=20, value=100)
    strength_picker.width = 100
    distance_picker.width = 100
    radius_picker.width = 100
    theta_picker.width = 100
    distance_max_picker.width = 100

    @pn.depends(strength_picker.param.value, distance_picker.param.value, radius_picker.param.value, theta_picker.param.value, distance_max_picker.param.value)
    def plot(strength, distance, radius, theta, distance_max):
        nodes = create_vega_nodes(graph)
        links = create_vega_links(graph)
        nodes_string = str(nodes).replace("'", '"')
        links_string = str(links).replace("'", '"')

        tooltip_signal = {}
        feature_labels = nodes[0].keys()
        for f in feature_labels:
            tooltip_signal[f] = "datum['" + f + "']"

        enter = {}
        if 'lens' in nodes[0]:
            enter = {
                "fill": {"field": "lens", "scale": "colour"},
                "tooltip": {"signal": str(tooltip_signal).replace('"','')} # Replace is necessary because no quotes allowed around "datum" (check vega editor)
            }
        else:
            enter = {
                "fill": {"value": "lightgrey"},
                "tooltip": {"signal": str(tooltip_signal).replace('"','')}
            }

        return pn.pane.Vega({
          "$schema": "https://vega.github.io/schema/vega/v5.json",
          "width": 1000,
          "height": 1000,
          "padding": 0,
          "autosize": "none",

          "signals": [
            { "name": "cx", "update": "width / 2" },
            { "name": "cy", "update": "height / 2" },
            {
              "description": "State variable for active node dragged status.",
              "name": "dragged", "value": 0,
              "on": [
                {
                  "events": "symbol:mouseout[!event.buttons], window:mouseup",
                  "update": "0"
                },
                {
                  "events": "symbol:mouseover",
                  "update": "dragged || 1"
                },
                {
                  "events": "[symbol:mousedown, window:mouseup] > window:mousemove!",
                  "update": "2", "force": True
                }
              ]
            },
            {
              "description": "Graph node most recently interacted with.",
              "name": "dragged_node", "value": None,
              "on": [
                {
                  "events": "symbol:mouseover",
                  "update": "dragged === 1 ? item() : dragged_node"
                }
              ]
            },
            {
              "description": "Flag to restart Force simulation upon data changes.",
              "name": "restart", "value": False,
              "on": [
                {"events": {"signal": "dragged"}, "update": "dragged > 1"}
              ]
            }
          ],
          "scales": [
            {
              "name": "colour",
              "type": "linear",
              "domain": {"data": "node-data", "field": "lens"},
              "range": {"scheme": "rainbow"}
            }
          ],

          "data": [
            {
              "name": "node-data",
              "values": nodes
            },
            {
              "name": "link-data",
              "values": links
            }
          ],

          "marks": [
            {
              "name": "nodes",
              "type": "symbol",
              "zindex": 1,

              "from": {"data": "node-data"},
              "on": [
                {
                  "trigger": "dragged",
                  "modify": "dragged_node",
                  "values": "dragged === 1 ? {fx:dragged_node.x, fy:dragged_node.y} : {fx:x(), fy:y()}"
                },
                {
                  "trigger": "!dragged",
                  "modify": "dragged_node", "values": "{fx: null, fy: null}"
                }
              ],

              "encode": {
                "enter": enter,
                "update": {
                  "size": {"value": 50},
                  "cursor": {"value": "pointer"}
                }
              },

              "transform": [
                {
                  "type": "force",
                  "iterations": 300,
                  "velocityDecay": 0.5,
                  "restart": {"signal": "restart"},
                  "static": False,
                  "forces": [
                    {"force": "center", "x": {"signal": "cx"}, "y": {"signal": "cy"}},
                    {"force": "collide", "radius": radius},
                    {"force": "nbody", "strength": strength, "theta": theta, "distanceMax": distance_max},
                    {"force": "link", "links": "link-data", "distance": distance}
                  ]
                }
              ]
            },
            {
              "type": "path",
              "from": {"data": "link-data"},
              "interactive": False,
              "encode": {
                "update": {
                  "stroke": {"value": "lightgrey"}
                }
              },
              "transform": [
                {
                  "type": "linkpath", "shape": "line",
                  "sourceX": "datum.source.x", "sourceY": "datum.source.y",
                  "targetX": "datum.target.x", "targetY": "datum.target.y"
                }
              ]
            }
          ]
        })

    return pn.Row(pn.Column(strength_picker, distance_picker, radius_picker, theta_picker, distance_max_picker), plot).show()

########
#### Evaluate result
########
def graph_to_distancematrix(graph):
    return graph.shortest_paths_dijkstra()

## Calculate correlation
def correlation_between_distance_matrices(matrix1, matrix2):
    '''
    correlation_between_distance_matrices(highD_dist_matrix, list(graph_to_distancematrix(mst)))}
    '''
    return np.corrcoef(matrix1.flatten(), np.asarray(matrix2).flatten())[0][1]

def create_list_of_all_links_with_values(highD_dist_matrix):
    '''
    This function creates the full list:
    [
        { from: 0,
          to: 1,
          highDd: 1.0 },
        { from: 0,
          to: 2,
          highDd: 2.236 },
        ...
    '''
    all_links = []
    l = len(highD_dist_matrix[0])
    for i in range(0,l):
        for j in range(i+1, l):
            all_links.append({
                'from': i,
                'to': j,
                'highDd': highD_dist_matrix[i][j]
            })
    return all_links

## Remove links that are already in MST
def create_list_of_links_to_add(list_of_links, graph):
    '''
    This (1) removes all MST links and
         (2) sorts all based on distance.
    IMPORTANT!!
    Possible links are sorted according to their distance in the original
    high-dimensional space, and _NOT_ based on the error in the distances
    between the high-dimensional space and the MST.
    Example: below is the dataset from data/sim.csv, with the MST indicated.
    If we sort by the _error_, then the first link that will be added is
    between the points indicated with an 'o' (because they lie at the opposite
    ends of the MST). If we sort by distance in the original high-D space, the
    first link that will be added is between the points that are indicated
    with a 'v'.

                *--*--*--*
                |        |
          *--o  *        *
          |     |        |
          *     *        *
          |     |        |
    *--*--*--v  v  o--*--*
       |        |
       *        *
       |        |
       *--*--*--*

    '''

    output = deepcopy(list_of_links)
    ## Remove the links that are already in the MST
    for e in graph.es():
        elements = list(filter(lambda x:x['to'] == e.target and x['from'] == e.source, list_of_links))
        output.remove(elements[0])
    ## Sort the links based on distance in original space
    output.sort(key = lambda x: x['highDd']) ## IMPORTANT!
    return output

## Add links to graph
def add_links_to_graph(graph, highD_dist_matrix, list_of_links_to_add, n):
    new_graph = deepcopy(graph)
    new_graph.add_edges(list(map(lambda x:(x['from'],x['to']), list_of_links_to_add[:int(n)])))
    distances = []
    for e in new_graph.es():
        distances.append(highD_dist_matrix[e.source][e.target])
    new_graph.es()['distance'] = distances
    return new_graph

## Using basinhopping
def cost_function(nr_of_links, args):
    graph = args['graph']
    list_of_links_to_add = args['list_of_links_to_add']
    highD_dist_matrix = args['highD_dist_matrix']

    new_graph = add_links_to_graph(graph, highD_dist_matrix, list_of_links_to_add, nr_of_links)
    return 1 - correlation_between_distance_matrices(highD_dist_matrix, list(graph_to_distancematrix(new_graph)))

def run_basinhopping(cf, mst, links_to_add, highD_dist_matrix, debug = False):
    '''
    Returns new graph.
        cf = cost_function
        start = start x
    '''
    disp = False
    if debug: disp = True
    start = len(mst.es())
    minimizer_kwargs = {'args':{'graph':mst,'list_of_links_to_add':links_to_add,'highD_dist_matrix':highD_dist_matrix}}
    result = optimize.basinhopping(
        cf,
        start,
        disp=disp,
        minimizer_kwargs=minimizer_kwargs
    )
    if debug:
        print(result)
    g = add_links_to_graph(mst, highD_dist_matrix, links_to_add, result.x[0])
    return g

########
#### Bringing everything together
########
def run_stad(highD_dist_matrix, lens=[], features={}, debug=False):
    '''
    Options:
    * `lens` needs to be an array with a single numerical value for each datapoint,
       or an empty array
    * `features` is a list of feature that will be added to the
       node tooltip. Format: {'label1': [value1, value2], 'label2': [value1, value2]}
    '''
    ## Check if distance matrix is normalised
    if ( np.min(highD_dist_matrix) < 0 or np.max(highD_dist_matrix) > 1 ):
        print("ERROR: input distance matrix needs to be normalised between 0 and 1")
        exit()

    if debug: print("Tweaking distance matrix if we're working with a lens")
    if len(lens) > 0:
        assigned_bins = assign_bins(lens, 5)
        dist_matrix = create_lensed_distmatrix_1step(highD_dist_matrix, assigned_bins)
    else:
        dist_matrix = highD_dist_matrix
    if debug: print("Calculating MST")
    mst = create_mst(dist_matrix, lens, features)
    if debug: print("Creating list of all links")
    all_links = create_list_of_all_links_with_values(dist_matrix)
    if debug: print("Removing MST links and sorting")
    list_of_links_to_add = create_list_of_links_to_add(all_links, mst)
    if debug: print("Start basinhopping")
    g = run_basinhopping(
            cost_function,
            mst,
            list_of_links_to_add,
            dist_matrix,
            debug=debug)
    return g

def main():
    data = pd.read_csv('data/five_circles.csv', header=0)
    values = data[['x','y']].values.tolist()
    lens = data['hue'].map(lambda x:hex_to_hsv(x)[0]).values
    xs = data['x'].values.tolist()
    ys = data['y'].values.tolist()
    hues = data['hue'].values.tolist()
    highD_dist_matrix = calculate_highD_dist_matrix(values)
    g = run_stad(highD_dist_matrix, lens=lens, features={'x':xs, 'y':ys, 'hue': hues})
    # g = run_stad(highD_dist_matrix, features={'x':xs, 'y':ys, 'hue':hues})
    draw_stad(g)

if __name__ == '__main__':
    main()
