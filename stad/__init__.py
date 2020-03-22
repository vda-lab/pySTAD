import click

import pandas as pd
import igraph as ig
import numpy as np
# from copy import deepcopy
from scipy import optimize
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree, floyd_warshall, bellman_ford, johnson, shortest_path
from math import exp
from numpy import log, log10
import datetime
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

# VISUALISATION FUNCTIONS

import panel as pn
import vega
pn.extension('vega')

from bokeh.plotting import figure, output_notebook, show
# from bokeh.layouts import row, column, gridplot
from bokeh.models import Span
output_notebook()


def plot_matrix(matrix, title):
    size = len(matrix)
    p = figure(title=title,
               x_range=(0, size), y_range=(0, size), plot_width=250, plot_height=250, toolbar_location=None, tooltips=[("index", "$index"),("x", "$x"), ("y", "$y")])
    p.image(image=[matrix], x=0, y=0, dw=size, dh=size, palette="Blues256")
    return p


def plot_trend(plot_type, xlabel, ylabel, xs, ys, title, line_position=None, line_orientation='vertical'):
    p = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel,
               plot_width=400, plot_height=300, toolbar_location=None, tooltips=[("index", "$index"),("x", "$x"), ("y", "$y")])
    if plot_type == 'line':
        p.line(xs, ys)
    else:
        p.scatter(xs,ys)
    if line_position:
        if line_orientation == 'vertical':
            dim = 'height'
        else:
            dim = 'width'
        line = Span(location=line_position, dimension=dim, line_color='red', line_width=1)
        p.add_layout(line)
    return p


def plot_trend_vega(d):
    return pn.pane.Vega({
      "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
      "width": 500,
      "height": 500,
      "data": {"values": d},
      "layer": [
        {
          "mark": {"type": "line", "tooltip": {"data": "content"}},
          "encoding": {
            "x": {"field": "x", "type": "quantitative", "title": "nr of links"},
            "y": {"field": "y", "type": "quantitative", "title": "correlation"},
            "opacity": {"value": 0.5}

          }
        },
        {
          "mark": {"type": "circle", "tooltip": {"data": "content"}},
          "encoding": {
            "x": {"field": "x", "type": "quantitative"},
            "y": {"field": "i", "type": "quantitative", "title": "epoch"},
            "color": {"field": "colour",
              "scale": {
                "domain": ["accepted", "still-accepted", "rejected"],
                "range": ["#4daf4a", "#ff7f00", "#e41a1c"]}},
            "opacity": {"value": 0.8}
          }
        }
      ],
      "resolve": {"scale": {"y": "independent"}}
    })

def plot_spearman(final_unit_adj, highD_dist_matrix):
    xs = np.triu(highD_dist_matrix, 1).flatten()
    graph_dist = shortest_path(final_unit_adj, method='D', directed=False, unweighted=True)
    ys = np.triu(graph_dist, 1).flatten()
    p = figure(title="Spearman comparison of distances", x_axis_label="Original distances", y_axis_label="Graph distances",
               plot_width=400, plot_height=300,
               toolbar_location=None, tooltips=[("index", "$index"),("x", "$x"), ("y", "$y")])
    p.scatter(xs, ys, fill_alpha=0.05, line_color=None)
    return p


def plot_qa(one_sided_mst,not_mst,final_unit_adj,final_weighted_adj,highD_dist_matrix,best_x,best_y,distances,xs,ys):
    delta_xs = []
    for i in range(1, len(xs)):
        delta_xs.append(xs[i]-xs[i-1])
    return pn.Column(
        pn.Row( plot_matrix(one_sided_mst, 'MST'),
                plot_matrix(not_mst, 'Not MST'),
                plot_matrix(final_unit_adj, 'Final matrix'),
                plot_matrix(final_weighted_adj, 'Final weighted matrix'),
                plot_matrix(highD_dist_matrix, 'Original matrix')),
        pn.Row( plot_trend('scatter','nr_of_links','correlation', xs,ys,"Correlation vs nr links", best_x, 'vertical'),
                plot_trend('scatter','iteration','correlation',range(0, len(ys)), ys, "Evolution of correlation (Should stabilise to maximum)", best_y, 'horizontal'),
                plot_trend('scatter','iteration','nr_of_links',range(0, len(xs)), xs, "Evolution of nr of links (Should stabilise)", best_x, 'horizontal')),
        pn.Row( plot_spearman(final_unit_adj, highD_dist_matrix),
                plot_trend('line','order','similarity',range(0, len(distances)),1-distances,"Ordered similarities of non-mst"),
                plot_trend('line','iteration','Change in nr_of_links between steps',range(0, len(delta_xs)), delta_xs, "Delta xs (Should stabilise around 0)")))


def create_vega_nodes(graph, features={}):
    nodes = []
    # The following will automatically pick up the lens...
    feature_labels = features.keys()
    for i in range(0, len(graph.vs())):
        node = {"name": i}
        nodes.append(node)
    for f in feature_labels:
        for i, node in enumerate(nodes):
            node[f] = features[f][i]
    return nodes


def create_vega_links(graph):
    output = []
    for e in graph.es():
        output.append({"source": e.source, "target": e.target, "value": e.attributes()['weight']})
    return output


def format_vertex_as_gephi(v, feature_labels):
    output = str(v['name'])
    for fl in feature_labels:
        output += "\t" + str(v[fl])
    output += "\n"
    return output


def create_gephi_files(graph, filename, features = {}):
    nodes = []
    feature_labels = features.keys()
    for i in range(0, len(graph.vs())):
        node = {"name": i}
        nodes.append(node)
    for f in feature_labels:
        for i, node in enumerate(nodes):
            node[f] = features[f][i]

    feature_labels = ['name'] + list(feature_labels)

    with open(filename + '_nodes.csv', 'w') as f:
        header = "id"
        for fl in feature_labels:
            header += "\t" + str(fl)
        header += "\n"
        f.write(header)

        # for v in graph.vs():
        for node in nodes:
            f.write(format_vertex_as_gephi(node, feature_labels))

    with open(filename + '_edges.csv', 'w') as f:
        f.write("source\ttarget\tvalue\n")
        for e in graph.es():
            f.write(str(e.source) + "\t" + str(e.target) + "\t" + str(e.attributes()['weight']) + "\n")


def draw_stad(graph, features={}):
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
        nodes = create_vega_nodes(graph, features)
        # print(nodes[0])
        links = create_vega_links(graph)
        # nodes_string = str(nodes).replace("'", '"')
        # links_string = str(links).replace("'", '"')

        tooltip_signal = {}
        feature_labels = nodes[0].keys()
        for f in feature_labels:
            tooltip_signal[f] = "datum['" + f + "']"

        enter = {}
        if 'colour' in nodes[0]:
            enter = {
                "fill": {"field": "colour"},
                "tooltip": {"signal": str(tooltip_signal).replace('"','')} # Replace is necessary because no quotes allowed around "datum" (check vega editor)
            }
        elif 'lens' in nodes[0]:
            enter = {
                "fill": {"field": "lens", "scale": "colour"},
                "tooltip": {"signal": str(tooltip_signal).replace('"', '')}
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

    return pn.Row(pn.Column(strength_picker, distance_picker, radius_picker, theta_picker, distance_max_picker), plot)


## UTIL FUNCTIONS
def calculate_highD_dist_matrix(data):
    non_normalized = euclidean_distances(data)
    # non_normalized = cosine_distances(data)
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


# LOAD DATASET


def flatten(l):
    return [item for sublist in l for item in sublist]


def load_testdata(dataset):
    if dataset == 'horse':
        data = pd.read_csv('stad/data/horse.csv', header=0)
        # sample_indices = [383, 100, 8, 375, 314, 253, 310, 317, 72, 160, 369, 240, 243, 207, 232, 106, 394, 446, 221, 94, 391, 158, 392, 421, 267, 498, 319, 75, 38, 123, 317, 293, 354, 363, 456, 151, 168, 117, 386, 445, 373, 305, 242, 251, 113, 353, 379, 133, 452, 355, 162, 495, 167, 207, 463, 209, 105, 68, 492, 147, 321, 380, 283, 193, 378, 160, 331, 199, 21, 451, 173, 404, 326, 144, 414, 381, 361, 337, 335, 80, 230, 482, 269, 475, 135, 368, 192, 408, 1, 36, 450, 495, 219, 94, 157, 176, 181, 70, 314, 273, 443, 106, 189, 91, 198, 176, 135, 253, 72, 260, 242, 439, 78, 127, 171, 370, 360, 396, 267, 435, 332, 341, 342, 358, 358, 448, 109, 323, 355, 424, 125, 21, 401, 206, 214, 427, 208, 15, 374, 197, 66, 407, 420, 19, 173, 363, 118, 454, 461, 269, 23, 327, 484, 447, 275, 301, 122, 32, 275, 395, 478, 28, 61, 181, 41, 238, 307, 290, 483, 44, 358, 276, 5, 297, 117, 153, 352, 247, 469, 163, 445, 65, 197, 467, 439, 240, 257, 203, 346, 123, 33, 54, 395, 263, 321, 497, 97, 444, 136, 452, 223, 175, 81, 33, 109, 49, 367, 404, 239, 140, 168, 387, 428, 26, 132, 44, 255, 480, 141, 461, 82, 104, 461, 237, 170, 136, 333, 220, 134, 78, 160, 289, 299, 182, 155, 376, 426, 2, 479, 145, 304, 251, 193, 322, 253, 487, 120, 169, 73, 65, 34, 208, 497, 291, 283, 416, 46, 163, 248, 185, 400, 385, 371, 277, 434, 81, 443, 30, 294, 467, 354, 256, 27, 253, 448, 388, 167, 342, 56, 46, 341, 246, 62, 366, 351, 274, 475, 87, 210, 174, 159, 352, 482, 326, 462, 39, 260, 107, 174, 450, 133, 351, 185, 86, 297, 447, 68, 288, 344, 41, 227, 183, 168, 57, 371, 306, 148, 252, 48, 43, 126, 8, 97, 466, 222, 141, 275, 184, 68, 65, 488, 384, 92, 475, 340, 28, 63, 343, 177, 290, 484, 205, 205, 323, 138, 264, 52, 169, 95, 30, 351, 114, 99, 204, 315, 188, 397, 226, 356, 148, 353, 14, 453, 165, 238, 419, 324, 105, 412, 447, 477, 56, 227, 447, 126, 54, 477, 409, 413, 42, 435, 482, 46, 475, 261, 465, 65, 338, 74, 381, 445, 344, 154, 475, 69, 373, 3, 210, 195, 284, 200, 322, 147, 35, 53, 346, 67, 207, 46, 211, 7, 462, 210, 194, 451, 46, 307, 292, 83, 115, 451, 46, 97, 492, 243, 52, 21, 233, 277, 2, 386, 336, 478, 5, 365, 471, 375, 64, 439, 477, 134, 335, 262, 88, 136, 103, 5, 219, 444, 319, 298, 353, 6, 316, 12, 243, 203, 41, 19, 406, 24, 343, 361, 362, 225, 162, 318, 17, 454, 83, 426, 422, 173, 11, 153, 158, 429, 31, 223, 489, 494, 297, 65, 55, 29, 366, 473, 24, 60, 249, 407, 391, 59, 67, 326, 100, 327, 444, 479, 39]
        # data = data.take(sample_indices)
        # data = data.sample(n=500)
        data['bin_x_10'] = pd.qcut(data.x, q=10, labels=range(0,10))
        data['bin_y_10'] = pd.qcut(data.y, q=10, labels=range(0, 10))
        data['bin_z_10'] = pd.qcut(data.z, q=10, labels=range(0, 10))

        sample = []
        for i in range(0,10):
            sample.append(list(map(lambda x:[x[0],x[1],x[2]], data[data.bin_y_10 == i].sample(30).values)))

        sample = flatten(sample)

        x_min = min(sample[0])
        x_max = max(sample[0])
        lens = list(map(lambda x: normalise_number_between_0_and_255(x[0], x_min, x_max), sample))
        features = {
            'lens': lens
        }
        # print(features)
        # lens = data[0].map(lambda x: normalise_number_between_0_and_255(x, x_min, x_max)).values

        highD_dist_matrix = calculate_highD_dist_matrix(sample)
        return highD_dist_matrix, lens, features
    elif dataset == 'simulated':
        data = pd.read_csv('stad/data/sim.csv', header=0)
        values = data[['x','y']]
        lens = np.zeros(1)
        highD_dist_matrix = calculate_highD_dist_matrix(values)
        return highD_dist_matrix, lens, {}
    elif dataset == 'circles':
        data = pd.read_csv('stad/data/five_circles.csv', header=0)
        values = data[['x','y']].values.tolist()
        lens = data['hue'].map(lambda x: hex_to_hsv(x)[0]).values
        features={
            'x': data['x'].values.tolist(),
            'y': data['y'].values.tolist(),
            'colour': data['hue'].values.tolist()
        }
        highD_dist_matrix = calculate_highD_dist_matrix(values)
        return (highD_dist_matrix, lens, features)
        # return (highD_dist_matrix, [], features)
    elif dataset == 'barcelona':
        non_normalised_highD_dist_matrix = np.array(pd.read_csv('stad/data/barcelona_distance_matrix.csv', header=None))
        max_value = np.max(non_normalised_highD_dist_matrix)
        highD_dist_matrix = non_normalised_highD_dist_matrix/max_value

        dates = np.array(pd.read_csv('stad/data/dates.csv', header=0))
        calendar_dates = list(map(lambda x:x[1], dates))
        days_of_week = list(map(lambda x:x[2], dates))
        colours = list(map(lambda x:x[3], dates))

        features={
            'dow': days_of_week,
            'date': calendar_dates,
            'colour': colours
        }

        return highD_dist_matrix, [], features
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


# LENS FUNCTIONS


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


# STAD


def create_mst(dist_matrix):
    """For a given distance matrix, returns the unit-distance MST."""
    # Use toarray because we want a regular nxn matrix, not the scipy sparse matrix.
    # np.random.seed(1)
    mst = minimum_spanning_tree(dist_matrix).toarray()
    # Set every edge weight to 1.
    mst = np.where(mst > 0, 1, 0)
    # Symmetrize.
    mst += mst.T
    mst = mst.astype('float32')
    return mst


def triu_mask(m, k=0):
    """
    For a given matrix m, returns like-sized boolean mask that is true for all
    elements `k` offset from the diagonal.
    """
    mask = np.zeros_like(m, dtype=np.bool)
    idx = np.triu_indices_from(m, k=k)
    mask[idx] = True
    return mask


def masked_edges(adj, mask):
    """
    For a given adjacency matrix and a like-sized boolean mask, returns a mask
    of the same size that is only true for the unique edges (the upper
    triangle). Assumes a symmetrical adjacency matrix.
    """
    return np.logical_and(triu_mask(adj, k=1), mask)


def ordered_edges(distances, mask):
    """
    For a given adjacency matrix with `n` edges and a like-sized mask, returns
    an n x 2 array of edges sorted by distance.
    """
    # We are only interested in the indices where our mask is truthy.
    # On a boolean array nonzero returns the true indices.
    # indices holds a tuple of arrays, one for each dimension.
    indices = np.nonzero(mask)
    ds = distances[indices]
    # argsort returns the sorted indices of the distances.
    # Note: these are not the same as the indices of our mask.
    order = np.argsort(ds)
    # We wish to return a single array, so we use `stack` to combine the two nx1
    # arrays into one nx2 array.
    combined_indices = np.stack(indices, 1)
    # Finally, we reorder our combined indices to be in the same order as the sorted distances.
    return [combined_indices[order].astype('int32'), ds[order]]


def add_unit_edges_to_matrix(adj_m, edges):
    """
    For a given adjacency matrix and an nx2 array of edges, returns a new
    adjacency matrix with the edges added. Symmetrizes the edges.
    """
    new_adj = adj_m.copy()
    for edge in edges:
        x = edge[0]
        y = edge[1]
        new_adj[x][y] = 1
        # new_adj[y][x] = 1
    return new_adj


# class MyTakeStep(object):
#     def __init__(self, initial_stepsize, initial_temperature, max_links=99999):
#         self.initial_stepsize = initial_stepsize
#         self.temperature = initial_temperature
#         self.max_links = max_links
#
#     def __call__(self, previous_nr_of_links):
#         new_stepsize = self.initial_stepsize*self.temperature
#         self.temperature = self.temperature*0.98
#         nr_of_links = int(np.random.normal(previous_nr_of_links, new_stepsize))
#         if nr_of_links < 0:
#             nr_of_links = 0
#         elif nr_of_links > self.max_links:
#             nr_of_links = self.max_links
#         return nr_of_links


def cost_function(nr_of_links, one_sided_mst, edges, highD_dist_matrix):
    adj = add_unit_edges_to_matrix(one_sided_mst, edges[:nr_of_links])
    dist = shortest_path(adj, method="D", directed=False, unweighted=True)
    corr = np.corrcoef(dist.flatten(), highD_dist_matrix.flatten())[0][1]
    return corr, dist


def take_step(previous_nr_of_links, max_links, temperature, more_links):
    stepsize = int(max_links * temperature)

    abs_random_jump_factor = np.abs(np.random.normal())
    if more_links:
        random_jump_factor = abs_random_jump_factor
    else:
        random_jump_factor = -abs_random_jump_factor

    nr_of_links = int(previous_nr_of_links + (random_jump_factor * stepsize))
    if nr_of_links < 0:
        nr_of_links = 0
    elif nr_of_links > max_links:
        nr_of_links = max_links

    return nr_of_links


def run_custom_basinhopping(one_sided_mst, not_mst, edges, highD_dist_matrix):
    # global xs, ys, dists
    xs = []
    ys = []
    decision_colours = []
    tmp_best_correlations = []
    directions = [] # true if more, false if less
    dists = []
    max_links = int(np.sum(not_mst))
    best_correlation = 0
    best_nr_of_links = 0

    tmp_best_correlations.append(best_correlation)
    directions.append(True)

    temperatures = []
    nr_iterations = 100
    for i in range(1, nr_iterations):
        temperatures.append((log10(nr_iterations)-log10(i))/log10(nr_iterations))
        # temperatures.append((5-log(i))/5)
    # temperatures = list(map(lambda x:(4.596-log(x))/4.596, range(1,100)))

    # previous_nr_of_links = 0
    previous_nr_of_links = int(max_links/10)
    for temperature in temperatures:
        # stepsize = int(max_links * temperature)
        # nr_of_links = take_step(0, max_links, stepsize)
        direction = directions[-1]
        nr_of_links = take_step(previous_nr_of_links, max_links, temperature, direction)
        previous_nr_of_links = nr_of_links
        # print("nr_of_links: " + str(nr_of_links))
        # print("correlation: " + str(best_correlation))

        new_corr, dist = cost_function(nr_of_links, one_sided_mst, edges, highD_dist_matrix)
        print(str(new_corr) + "\t" + str(best_correlation))
        if new_corr > best_correlation:
            print("  => ACCEPTED")
            best_correlation = new_corr
            best_nr_of_links = nr_of_links
            decision_colours.append('accepted')
            directions.append(directions[-1])

        else:
            cutoff = exp((new_corr-best_correlation)/temperature)
            random_number = np.random.random()
            # random_number = 0.3
            print("accept if cutoff > random_number?: " + str(cutoff) + ' > ' + str(random_number))
            # if cutoff > random_number:
            if cutoff < random_number:
                print("  => STILL ACCEPTED")
                directions.append(directions[-1])
                best_correlation = new_corr
                best_nr_of_links = nr_of_links
                decision_colours.append('still-accepted')
            else:
                print("  => REJECTED")
                directions.append(not directions[-1])
                decision_colours.append('rejected')

        xs.append(nr_of_links)
        ys.append(new_corr)
        dists.append(dist)
        tmp_best_correlations.append(best_correlation)

    return best_nr_of_links, best_correlation, xs, ys, dists, decision_colours, tmp_best_correlations, directions


# def run_basinhopping(one_sided_mst, not_mst, edges, highD_dist_matrix):
#     global xs, ys, dists
#     xs = []
#     ys = []
#     dists = []
#     initial_temperature = 1
#     min_links = int(np.sum(one_sided_mst))
#     max_links = int(np.sum(not_mst))
#     initial_stepsize = int((max_links - min_links) / 100)
#
#     def cost_function(nr_of_links):
#         nr_of_links = int(nr_of_links)
#         adj = add_unit_edges_to_matrix(one_sided_mst, edges[:nr_of_links])
#         dist = shortest_path(adj, method='D', directed=False, unweighted=True)
#         #         dist = dijkstra(adj, directed=False, unweighted=True)
#         #         dist = bellman_ford(adj, directed=False)
#         #         dist = johnson(adj, directed=False)
#         #         dist = floyd_warshall(adj, directed=False)
#         result = np.corrcoef(dist.flatten(), highD_dist_matrix.flatten())[0][1]
#         xs.append(nr_of_links)
#         ys.append(result)
#         dists.append(dist)
#         return 1 - result
#
#     #     minimizer_kwargs = {'args':{"one_sided_mst": one_sided_mst}}
#     minimizer_kwargs = {'method': 'L-BFGS-B'}
#     result = optimize.basinhopping(
#         cost_function,
#         min_links,
#         disp=False,
#         #                     T=temperature,
#         #                     stepsize=stepsize,
#         #                     interval=100,
#         #                     niter=1000,
#         minimizer_kwargs=minimizer_kwargs,
#         take_step=MyTakeStep(initial_stepsize=initial_stepsize, initial_temperature=initial_temperature,
#                              max_links=max_links)
#     )
#     return [result, xs, ys, dists]


def create_final_unit_adj(one_sided_mst, edges, nr_of_links_to_add):
    return add_unit_edges_to_matrix(one_sided_mst, edges[:nr_of_links_to_add])


# RUN STAD
@click.command()
@click.argument('dataset', type=click.Choice(['circles', 'horse', 'simulated', 'barcelona'], case_sensitive=False))
@click.option('--nr_bins', default=5)
@click.option('--lens/--no-lens', 'use_lens', default=False)
def main(dataset, nr_bins, use_lens):
    start = datetime.datetime.now()
    original_highD_dist_matrix, lens_data, features = load_testdata(dataset)

    if use_lens and len(lens_data) > 0:
        assigned_bins = assign_bins(lens_data, nr_bins)
        highD_dist_matrix = create_lensed_distmatrix_1step(original_highD_dist_matrix, assigned_bins)
    else:
        highD_dist_matrix = original_highD_dist_matrix

    mst = create_mst(highD_dist_matrix)
    one_sided_mst = np.where(triu_mask(mst, k=1), mst, 0)
    not_mst = masked_edges(highD_dist_matrix, mst == 0)
    edges, distances = ordered_edges(highD_dist_matrix, not_mst)  # We'll need distances later for STAD-R
    best_nr_of_links, best_correlation, xs, ys, dists, decision_colours, tmp_best_correlations, directions = run_custom_basinhopping(one_sided_mst, not_mst, edges, original_highD_dist_matrix)
    final_unit_adj = create_final_unit_adj(one_sided_mst, edges, best_nr_of_links)
    # result, xs, ys, dists = run_basinhopping(one_sided_mst, not_mst, edges, highD_dist_matrix)
    # final_unit_adj = create_final_unit_adj(one_sided_mst, edges, int(result['x'][0]))
    final_weighted_adj = np.where(final_unit_adj == 1, original_highD_dist_matrix, 0)
    # final_weighted_adj = np.where(one_sided_mst == 1, original_highD_dist_matrix, 0)
    best_x = best_nr_of_links
    best_y = best_correlation
    print("Number of tested positions: " + str(len(xs)))
    print("Number of links in mst: " + str(np.sum(one_sided_mst)))
    print("Number of links added: " + str(best_x))
    print("Number of links final: " + str(np.sum(final_unit_adj)))
    print("Final correlation: " + str(best_y))
    g = ig.Graph.Weighted_Adjacency(list(final_weighted_adj))

    stop = datetime.datetime.now()
    time_delta = stop - start
    print(time_delta)

    # create_gephi_files(g, dataset, features)

    my_d = []
    counter = 0
    for d in zip(xs,ys,decision_colours, tmp_best_correlations, directions):
        new_d = {}
        new_d['x'] = d[0]
        new_d['y'] = d[1]
        new_d['i'] = counter
        new_d['colour'] = d[2]
        new_d['tmp_best_correlation'] = d[3]
        new_d['more_links'] = d[4]
        my_d.append(new_d)
        counter += 1

    pn.Column(
        "# QA plots for " + dataset,
        "Number of tested positions: " + str(len(xs)) + "<br/>" +
        "Number of links in mst: " + str(np.sum(one_sided_mst)) + "<br/>" +
        "Number of links added: " + str(best_x) + "<br/>" +
        "Number of links final: " + str(np.sum(final_unit_adj)) + "<br/>" +
        "Final correlation: <b>" + "{:.2f}".format(best_y) + "</b>",
        plot_trend_vega(my_d),
        plot_qa(one_sided_mst,not_mst,final_unit_adj,final_weighted_adj,highD_dist_matrix,best_x,best_y,distances,xs,ys),
        draw_stad(g, features)
    ).show()


if __name__ == '__main__':
    main()
