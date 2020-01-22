---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
# !pip install git+https://github.com/holoviz/panel.git
```

```python
from IPython.display import display
```

```python
import pandas as pd
import igraph as ig
import numpy as np
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
```

```python
import panel as pn
import vega
pn.extension('vega')
```

# Functions only for testing

```python code_folding=[0]
def hex_to_number(hex):
    h = tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    return h[0]
```

# Create MST

```python code_folding=[0]
def calculate_highD_dist_matrix(data):
    return euclidean_distances(data)
```

```python code_folding=[0]
def matrix_to_topright_array(matrix):
    for i, vector in enumerate(highD_dist_matrix):
        for j, value in enumerate(vector):
            if ( j > i ):
                yield value
```

```python code_folding=[0]
def matrix_to_all_combinations(matrix):
    for i, vector in enumerate(highD_dist_matrix):
        for j, value in enumerate(vector):
            if ( j > i ):
                yield [i,j]
```

```python
data = pd.read_csv('data/five_circles.csv', header=0)
values = data[['x','y']]
colours = data['hue'].map(lambda x:hex_to_number(x))
highD_dist_matrix = calculate_highD_dist_matrix(values)
# print(len(list(matrix_to_topright_array(highD_dist_matrix))))
# print(list(matrix_to_all_combinations(highD_dist_matrix)))
```

```python code_folding=[0]
def create_mst(dist_matrix):
    complete_graph = ig.Graph.Full(len(dist_matrix[0]))
    complete_graph.vs["colour"] = colours
    complete_graph.es["distance"] = list(matrix_to_topright_array(dist_matrix))
    return complete_graph.spanning_tree(weights = list(matrix_to_topright_array(dist_matrix)))
```

```python
mst = create_mst(highD_dist_matrix)
```

## Draw the graph

```python code_folding=[0]
def create_vega_nodes(graph):
    for v in graph.vs():
        yield({"name": v.index, "colour": v.attributes()['colour']})

def create_vega_links(graph):
    for e in graph.es():
        yield({"source": e.source, "target": e.target, "value": e.attributes()['distance']})
```

```python
# nodes = list(create_vega_nodes(mst))
# links = list(create_vega_links(mst))
# nodes_string = str(nodes).replace("'", '"')
# links_string = str(links).replace("'", '"')
```

```python code_folding=[]
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
                  "fill": {"field": "colour", "scale": "colour"},
                  "tooltip": {"field": "name"}
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

    return pn.Row(pn.Column(strength_picker, distance_picker, radius_picker, theta_picker, distance_max_picker), plot).show()

# draw_stad(mst)
```

# Graph to distance matrix

```python code_folding=[0]
def graph_to_distancematrix(graph):
    return graph.shortest_paths_dijkstra()
```

# Calculate correlation

```python code_folding=[]
def correlation_between_distance_matrices(matrix1, matrix2):
    '''
    correlation_between_distance_matrices(highD_dist_matrix, list(graph_to_distancematrix(mst)))}
    '''
    return np.corrcoef(matrix1.flatten(), np.asarray(matrix2).flatten())[0][1]
# correlation_between_distance_matrices(highD_dist_matrix, list(graph_to_distancematrix(mst)))
```

# Create list of links to add
We need to have a list of all combinations of nodes (i.e. all possible links) with an indication on how wrong each is. For this we'll normalise the 2 matrices, substract them and see which node-pairs are most wrong.

```python code_folding=[0]
def normalise_matrix(mtx):
    '''
    Make all values in a matrix to be between 0 and 1
    '''
    return (mtx - np.min(mtx))/np.ptp(mtx)
```

```python code_folding=[0]
def identify_error_in_distances(mtx1, mtx2):
    '''
    Substract distance matrices. Should be used to substract
    the original distance matrix from the graph distance matrix.
    '''
    return normalise_matrix(mtx1) - normalise_matrix(mtx2)

# identify_error_in_distances(highD_dist_matrix, graph_to_distancematrix(mst))
```

```python code_folding=[0]
def create_list_of_all_links_with_value(error_matrix):
    links_to_add = []
    for i, vector in enumerate(error_matrix):
        for j, value in enumerate(vector):
            if ( j > i ):
                links_to_add.append([i,j,abs(value)])
    return links_to_add

# create_list_of_all_links_with_value(
#     identify_error_in_distances(
#         highD_dist_matrix,
#         graph_to_distancematrix(mst)))
```

## Remove links that are already in MST

```python
unsorted_links = create_list_of_all_links_with_value(identify_error_in_distances(highD_dist_matrix, graph_to_distancematrix(mst)))
```

```python code_folding=[]
def remove_mst_links_from_list(list_of_links, graph):
## SHOULD BE QUICKER BUT IS WRONG
## (WANT TO LOOP LESS)
#     output = deepcopy(list_of_links)
#     all_targets = []
#     for e in graph.es():
#         all_targets.append(e.target)
#     unique_targets = list(set(all_targets))
    
#     for t in unique_targets:
#         edges_with_target = list(filter(lambda x:x[1] == t, list_of_links))
#         for e in edges_with_target:
#             for l in list(filter(lambda x:x[0] == e, edges_with_target)):
#                 output.remove(l)
#     return output

# OR SIMPLER BUT SLOWER
    output = deepcopy(list_of_links)
    for e in graph.es():
        a = list(filter(lambda x:x[1] == e.target, list_of_links))
        b = list(filter(lambda x:x[0] == e.source, a))
        output.remove(b[0])
    return output
```

```python
unsorted_links_to_add_without_mst = remove_mst_links_from_list(unsorted_links, mst)
```

## Sort links

```python
sorted_links_to_add = deepcopy(unsorted_links_to_add_without_mst)
sorted_links_to_add.sort(key = lambda x: x[2], reverse=True)
```

# Add links to graph

```python code_folding=[]
def add_links_to_graph(list_of_links_to_add, n):
    new_graph = deepcopy(mst)
    new_graph.add_edges(list(map(lambda x:(x[0],x[1]), list_of_links_to_add[:n])))
    return new_graph
```

```python
g20 = add_links_to_graph(sorted_links_to_add, 20)
draw_stad(g30)
```

```python
g30 = add_links_to_graph(sorted_links_to_add, 30)
draw_stad(g30)
```

```python

```

```python

```
