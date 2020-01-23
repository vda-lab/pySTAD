import pandas as pd
import igraph as ig
import numpy as np
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy import optimize

import panel as pn
import vega
pn.extension('vega')

## Functions only for testing

def hex_to_number(hex):
    h = tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    return h[0]

def normalise_number_between_0_and_1(nr, domain_min, domain_max):
    return (nr-domain_min)/(domain_max - domain_min)

def normalise_number_between_0_and_255(nr, domain_min, domain_max):
    return 255*normalise_number_between_0_and_1(nr, domain_min, domain_max)

## Load data
def load_data(dataset):
    if dataset == 'horse':
        data = pd.read_csv('data/horse.csv', header=0)
        data = data.sample(n=1000)
        values = data[['x','y','z']].values.tolist()
        x_min = min(data['x'])
        x_max = max(data['x'])
        # zs = data['z'].values
        colours = data['x'].map(lambda x:normalise_number_between_0_and_255(x, x_min, x_max)).values
        return(values, colours)
    elif dataset == 'simulated':
        data = pd.read_csv('data/sim.csv', header=0)
        values = data[['x','y']]
        colours = np.zeros(1)
        return(values, colours)
    elif dataset == 'circles':
        data = pd.read_csv('data/five_circles.csv', header=0)
        values = data[['x','y']].values.tolist()
        colours = list(data['hue'].map(lambda x:hex_to_number(x)))
        return(values, colours)
    else:
        print("Dataset not known")

def calculate_highD_dist_matrix(data):
    return euclidean_distances(data)

## Create MST
def matrix_to_topright_array(matrix):
    for i, vector in enumerate(matrix):
        for j, value in enumerate(vector):
            if ( j > i ):
                yield value

def matrix_to_all_combinations(matrix):
    for i, vector in enumerate(highD_dist_matrix):
        for j, value in enumerate(vector):
            if ( j > i ):
                yield [i,j]

def create_mst(dist_matrix, colours):
    complete_graph = ig.Graph.Full(len(dist_matrix[0]))
    complete_graph.vs["colour"] = colours
    complete_graph.es["distance"] = list(matrix_to_topright_array(dist_matrix))
    return complete_graph.spanning_tree(weights = list(matrix_to_topright_array(dist_matrix)))

## Draw the graph
def create_vega_nodes(graph):
    for v in graph.vs():
        yield({"name": v.index, "colour": v.attributes()['colour']})

def create_vega_links(graph):
    for e in graph.es():
        yield({"source": e.source, "target": e.target, "value": e.attributes()['distance']})

def create_gephi_files(graph, filename):
    with open(filename + '_nodes.csv', 'w') as f:
        f.write("colour\n")
        counter = 0
        for v in graph.vs():
            f.write(str(v.attributes()['colour']) + "\n")
            counter += 1
    with open(filename + '_edges.csv', 'w') as f:
        f.write("source\ttarget\tvalue\n")
        for e in graph.es():
            f.write(str(e.source) + "\t" + str(e.target) + "\t" + str(e.attributes()['distance']) + "\n")

def draw_stad(dataset, graph):
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
        nodes = list(create_vega_nodes(graph))
        links = list(create_vega_links(graph))
        nodes_string = str(nodes).replace("'", '"')
        links_string = str(links).replace("'", '"')
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
              "domain": {"data": "node-data", "field": "colour"},
              "range": {"scheme": "oranges"}
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
                "enter": {
                  "fill": {"field": "colour", "scale": "colour"}
                },
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

    return pn.Row(pn.Column(pn.pane.Markdown("## Dataset: " + dataset),strength_picker, distance_picker, radius_picker, theta_picker, distance_max_picker), plot).show()


## Graph to distance matrix
def graph_to_distancematrix(graph):
    return graph.shortest_paths_dijkstra()

## Calculate correlation
def correlation_between_distance_matrices(matrix1, matrix2):
    '''
    correlation_between_distance_matrices(highD_dist_matrix, list(graph_to_distancematrix(mst)))}
    '''
    return np.corrcoef(matrix1.flatten(), np.asarray(matrix2).flatten())[0][1]

## Create list of links to add
def normalise_matrix(mtx):
    '''
    Make all values in a matrix to be between 0 and 1
    '''
    return (mtx - np.min(mtx))/np.ptp(mtx)

def identify_error_in_distances(mtx1, mtx2):
    '''
    Substract distance matrices. Should be used to substract
    the original distance matrix from the graph distance matrix.
    '''
    return normalise_matrix(mtx1) - normalise_matrix(mtx2)

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
    This removes all MST links and sorts all based on distance
    '''
    output = deepcopy(list_of_links)
    for e in graph.es():
        elements = list(filter(lambda x:x['to'] == e.target and x['from'] == e.source, list_of_links))
        output.remove(elements[0])
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

## Bringing everything together
def run_stad(values, colours, debug=False):
    if debug: print("Calculating highD distance matrix")
    highD_dist_matrix = calculate_highD_dist_matrix(values)
    if debug: print("Calculating MST")
    mst = create_mst(highD_dist_matrix, colours)
    if debug: print("Creating list of all links")
    all_links = create_list_of_all_links_with_values(highD_dist_matrix)
    if debug: print("Removing MST links and sorting")
    list_of_links_to_add = create_list_of_links_to_add(all_links, mst)

    g = run_basinhopping(
            cost_function,
            mst,
            list_of_links_to_add,
            highD_dist_matrix,
            debug=debug)
    return g

def main():
    dataset = 'circles'
    # dataset = 'horse'
    # dataset = 'simulated'
    values, colours = load_data(dataset)
    g = run_stad(values, colours, debug=True)
    draw_stad(dataset, g)
    # create_gephi_files(g, dataset)

if __name__ == '__main__':
    main()
