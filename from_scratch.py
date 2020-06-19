import click

import pandas as pd
import igraph as ig
import numpy as np
from copy import deepcopy
from scipy import optimize
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree, floyd_warshall, bellman_ford, johnson, shortest_path
from math import exp
from numpy import log

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
                plot_trend('line','iteration','correlation',range(0, len(ys)), ys, "Evolution of correlation (Should stabilise to maximum)", best_y, 'horizontal'),
                plot_trend('line','iteration','nr_of_links',range(0, len(xs)), xs, "Evolution of nr of links (Should stabilise)", best_x, 'horizontal')),
        pn.Row( plot_spearman(final_unit_adj, highD_dist_matrix),
                plot_trend('line','order','similarity',range(0, len(distances)),1-distances,"Ordered similarities of non-mst"),
                plot_trend('line','iteration','Change in nr_of_links between steps',range(0, len(delta_xs)), delta_xs, "Delta xs (Should stabilise around 0)")))


def create_vega_nodes(graph, lens=[], features={}):
    nodes = []
    # The following will automatically pick up the lens...
    feature_labels = features.keys()
    print("DEBUG: nr of nodes = " + str(len(graph.vs())))
    print(features)
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


def format_vertex_as_gephi(v, counter, has_lens=False, feature_labels=[]):
    output = str(counter)
    if has_lens:
        output += "\t" + str(v.attributes()['lens'])
    for fl in feature_labels:
        output += "\t" + str(v.attributes()[fl])
    output += "\n"
    return output


def create_gephi_files(graph, filename):
    feature_labels = graph.vs()[0].attributes().keys()
    has_lens = False
    if 'lens' in graph.vs()[0].attributes():
        has_lens = True

    with open(filename + '_nodes.csv', 'w') as f:
        header = "id"
        if has_lens:
            header += "\tlens"
        for fl in feature_labels:
            header += "\t" + str(fl)
        header += "\n"
        f.write(header)

        counter = 0
        for v in graph.vs():
            f.write(format_vertex_as_gephi(v,counter,has_lens,feature_labels))
            counter += 1

    with open(filename + '_edges.csv', 'w') as f:
        f.write("source\ttarget\tvalue\n")
        for e in graph.es():
            f.write(str(e.source) + "\t" + str(e.target) + "\t" + str(e.attributes()['weight']) + "\n")


def draw_stad(graph, lens=[], features={}):
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
        nodes = create_vega_nodes(graph, lens, features)
        print(nodes[0])
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


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


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


# LOAD DATASET


def load_testdata(dataset):
    if dataset == 'horse':
        data = pd.read_csv('stad/data/horse.csv', header=0)
        data = data.sample(n=500)
        values = data[['x','y','z']].values.tolist()
        x_min = min(data['x'])
        x_max = max(data['x'])
        # zs = data['z'].values
        lens = data['x'].map(lambda x: normalise_number_between_0_and_255(x, x_min, x_max)).values
        highD_dist_matrix = calculate_highD_dist_matrix(values)
        return highD_dist_matrix, lens, {}
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


# STAD


def create_mst(dist_matrix):
    """For a given distance matrix, returns the unit-distance MST."""
    # Use toarray because we want a regular nxn matrix, not the scipy sparse matrix.
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
    # We are only interested in the indicies where our mask is truthy.
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


class MyTakeStep(object):
    def __init__(self, initial_stepsize, initial_temperature, max_links=99999):
        self.initial_stepsize = initial_stepsize
        self.temperature = initial_temperature
        self.max_links = max_links

    def __call__(self, previous_nr_of_links):
        new_stepsize = self.initial_stepsize*self.temperature
        self.temperature = self.temperature*0.95
        nr_of_links = int(np.random.normal(previous_nr_of_links, new_stepsize))
        if nr_of_links < 0:
            nr_of_links = 0
        elif nr_of_links > self.max_links:
            nr_of_links = self.max_links
        return nr_of_links

def cost_function(nr_of_links, one_sided_mst, edges, highD_dist_matrix):
    nr_of_links = int(nr_of_links)
    adj = add_unit_edges_to_matrix(one_sided_mst, edges[:nr_of_links])
    dist = shortest_path(adj, method="D", directed=False, unweighted=True)
    corr = np.corrcoef(dist.flatten(), highD_dist_matrix.flatten())[0][1]
    return corr, dist


def take_step(previous_nr_of_links, max_links, stepsize):
    nr_of_links = int(np.random.normal(previous_nr_of_links, stepsize))
    if nr_of_links < 0:
        nr_of_links = 0
    elif nr_of_links > max_links:
        nr_of_links = max_links
    return nr_of_links


def run_custom_basinhopping(one_sided_mst, not_mst, edges, highD_dist_matrix):
    # global xs, ys, dists
    xs = []
    ys = []
    dists = []
    max_links = int(np.sum(not_mst))
    best_correlation = 0
    best_nr_of_links = 0

    temperatures = list(map(lambda x:(4.596-log(x))/4.596, range(1,100)))

    for temperature in temperatures:
        print("-----------")
        stepsize = int(max_links * temperature)
        print("temperature: " + str(temperature))
        print("stepsize: " + str(stepsize))
        nr_of_links = take_step(0, max_links, stepsize)
        print("nr_of_links: " + str(nr_of_links))
        print("correlation: " + str(best_correlation))

        new_corr, dist = cost_function(nr_of_links, one_sided_mst, edges, highD_dist_matrix)
        print("new correlation: " + str(new_corr))
        if ( new_corr > best_correlation ):
            print("  => ACCEPTED")
            best_correlation = new_corr
            best_nr_of_links = nr_of_links
        else:
            cutoff = exp((new_corr-best_correlation)/temperature)
            random_number = np.random.random()
            print("accept als cutoff > random_number?: " + str(cutoff) + ' > ' + str(random_number))
            if cutoff > random_number:
                print("  => STILL ACCEPTED")
                best_correlation = new_corr
                best_nr_of_links = nr_of_links
            else:
                print("  => REJECTED")

        xs.append(nr_of_links)
        ys.append(new_corr)
        dists.append(dist)

    return best_nr_of_links, best_correlation, xs, ys, dists


def run_basinhopping(one_sided_mst, not_mst, edges, highD_dist_matrix):
    global xs, ys, dists
    xs = []
    ys = []
    dists = []
    initial_temperature = 1
    min_links = int(np.sum(one_sided_mst))
    max_links = int(np.sum(not_mst))
    initial_stepsize = int((max_links - min_links) / 100)

    def cost_function(nr_of_links):
        nr_of_links = int(nr_of_links)
        adj = add_unit_edges_to_matrix(one_sided_mst, edges[:nr_of_links])
        dist = shortest_path(adj, method='D', directed=False, unweighted=True)
        #         dist = dijkstra(adj, directed=False, unweighted=True)
        #         dist = bellman_ford(adj, directed=False)
        #         dist = johnson(adj, directed=False)
        #         dist = floyd_warshall(adj, directed=False)
        result = np.corrcoef(dist.flatten(), highD_dist_matrix.flatten())[0][1]
        xs.append(nr_of_links)
        ys.append(result)
        dists.append(dist)
        return 1 - result

    #     minimizer_kwargs = {'args':{"one_sided_mst": one_sided_mst}}
    minimizer_kwargs = {'method': 'L-BFGS-B'}
    result = optimize.basinhopping(
        cost_function,
        min_links,
        disp=False,
        #                     T=temperature,
        #                     stepsize=stepsize,
        #                     interval=100,
        #                     niter=1000,
        minimizer_kwargs=minimizer_kwargs,
        take_step=MyTakeStep(initial_stepsize=initial_stepsize, initial_temperature=initial_temperature,
                             max_links=max_links)
    )
    return [result, xs, ys, dists]


def create_final_unit_adj(one_sided_mst, edges, nr_of_links_to_add):
    return add_unit_edges_to_matrix(one_sided_mst, edges[:nr_of_links_to_add])


# RUN STAD


@click.command()
@click.argument('dataset', type=click.Choice(['circles', 'horse', 'simulated', 'barcelona'], case_sensitive=False))
def main(dataset):
    highD_dist_matrix, lens_data, features = load_testdata(dataset)
    mst = create_mst(highD_dist_matrix)
    one_sided_mst = np.where(triu_mask(mst, k=1), mst, 0)
    not_mst = masked_edges(highD_dist_matrix, mst == 0)
    edges, distances = ordered_edges(highD_dist_matrix, not_mst)  # We'll need distances later for STAD-R
    best_nr_of_links, best_correlation, xs, ys, dists = run_custom_basinhopping(one_sided_mst, not_mst, edges, highD_dist_matrix)
    final_unit_adj = create_final_unit_adj(one_sided_mst, edges, best_nr_of_links)
    # result, xs, ys, dists = run_basinhopping(one_sided_mst, not_mst, edges, highD_dist_matrix)
    # final_unit_adj = create_final_unit_adj(one_sided_mst, edges, int(result['x'][0]))
    final_weighted_adj = np.where(final_unit_adj == 1, highD_dist_matrix, 0)
    best_x = best_nr_of_links
    best_y = best_correlation
    print("Number of tested positions: " + str(len(xs)))
    print("Number of links in mst: " + str(np.sum(one_sided_mst)))
    print("Number of links added: " + str(best_x))
    print("Number of links final: " + str(np.sum(final_unit_adj)))
    print("Final correlation: " + str(best_y))
    g = ig.Graph.Weighted_Adjacency(list(final_weighted_adj))

    pn.Column(
        "# QA plots for " + dataset,
        "Number of tested positions: " + str(len(xs)) + "<br/>" +
        "Number of links in mst: " + str(np.sum(one_sided_mst)) + "<br/>" +
        "Number of links added: " + str(best_x) + "<br/>" +
        "Number of links final: " + str(np.sum(final_unit_adj)) + "<br/>" +
        "Final correlation: " + str(best_y),
        plot_qa(one_sided_mst,not_mst,final_unit_adj,final_weighted_adj,highD_dist_matrix,best_x,best_y,distances,xs,ys),
        draw_stad(g, lens_data, features)
    ).show()

if __name__ == '__main__':
    main()
