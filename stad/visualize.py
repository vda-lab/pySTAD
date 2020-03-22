import numpy as np
from scipy.sparse.csgraph import shortest_path
import panel as pn
pn.extension('vega')

from bokeh.plotting import figure, output_notebook
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
            "y": {"field": "epoch", "type": "quantitative", "title": "epoch"},
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
