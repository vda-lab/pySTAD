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

# Load data

```python
def normalise_number_between_0_and_1(nr, domain_min, domain_max):
    return (nr-domain_min)/(domain_max - domain_min)

def normalise_number_between_0_and_255(nr, domain_min, domain_max):
    return 255*normalise_number_between_0_and_1(nr, domain_min, domain_max)
```

```python code_folding=[]
def calculate_highD_dist_matrix(data):
    return euclidean_distances(data)
```

```python
# ## horse
# data = pd.read_csv('data/horse.csv', header=0)
# data = data.sample(n=300)
# values = data[['x','y','z']].values.tolist()
# x_min = min(data['x'])
# x_max = max(data['x'])
# xs = data['x'].values
# ys = data['y'].values
# zs = data['z'].values
# colours = data['x'].map(lambda x:normalise_number_between_0_and_255(x, x_min, x_max)).values
# highD_dist_matrix = calculate_highD_dist_matrix(values)

# simulated
data = pd.read_csv('data/sim.csv', header=0)
values = data[['x','y']]
ids = data['id'].values
# colours = data['hue'].map(lambda x:hex_to_number(x))
highD_dist_matrix = calculate_highD_dist_matrix(values)

# ## five circles
# data = pd.read_csv('data/five_circles.csv', header=0)
# values = data[['x','y']]
# colours = data['hue'].map(lambda x:hex_to_number(x))
# highD_dist_matrix = calculate_highD_dist_matrix(values)
# print(len(list(matrix_to_topright_array(highD_dist_matrix))))
# print(list(matrix_to_all_combinations(highD_dist_matrix)))
```

# Create MST

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

```python code_folding=[]
def create_mst(dist_matrix):
    complete_graph = ig.Graph.Full(len(dist_matrix[0]))
    complete_graph.es["distance"] = list(matrix_to_topright_array(dist_matrix))
    return complete_graph.spanning_tree(weights = list(matrix_to_topright_array(dist_matrix)))
```

```python
mst = create_mst(highD_dist_matrix)
```

## Draw the graph

```python code_folding=[0, 4, 8]
def create_vega_nodes(graph):
    for v in graph.vs():
        yield({"name": v.index, "id": v.attributes()['id']})

def create_vega_links(graph):
    for e in graph.es():
        yield({"source": e.source, "target": e.target, "value": e.attributes()['distance']})

def create_gephi_files(graph, filename):
    with open(filename + '_nodes.csv', 'w') as f:
        f.write("id\tx\ty\n")
        counter = 0
        for v in graph.vs():
            f.write(str(v.attributes()['id']) + "\t" + str(v.attributes()['x']) + "\t" + str(v.attributes()['y']) + "\n")
            counter += 1
    with open(filename + '_edges.csv', 'w') as f:
        f.write("source\ttarget\tvalue\n")
        for e in graph.es():
            f.write(str(e.source) + "\t" + str(e.target) + "\t" + str(e.attributes()['distance']) + "\n")
```

```python code_folding=[0]
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
                  "fill": {"value": "lightgrey"},
                  "tooltip": {"field": "id"}
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

```python code_folding=[0]
def correlation_between_distance_matrices(matrix1, matrix2):
    '''
    correlation_between_distance_matrices(highD_dist_matrix, list(graph_to_distancematrix(mst)))}
    '''
    return np.corrcoef(matrix1.flatten(), np.asarray(matrix2).flatten())[0][1]
```

# Create list of links to add

We'll store everything into an array of dictionaries with these keys: `from` (row in the original distance matrix), `to` (column in the original distance matrix), `highDd` (distance in high-dimensional space), `graphd` (graph distance), `e` (error). E.g.

```
[
    { from: 0,
      to: 1,
      highDd: 1.0,
      graphd: 1,
      e: 0 },
    { from: 0,
      to: 2,
      highDd: 2.236,
      graphd: 2,
      e: 0.236 },
    ...
```

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

```python code_folding=[]
def create_list_of_all_links_with_values(highD_dist_matrix, graph, error_matrix):
    '''
    This function creates the full list: 
    [
        { from: 0,
          to: 1,
          highDd: 1.0,
          graphd: 1,
          e: 0 },
        { from: 0,
          to: 2,
          highDd: 2.236,
          graphd: 2,
          e: 0.236 },
        ...
    '''
    graph_distances = graph_to_distancematrix(graph)
    all_links = []
    l = len(highD_dist_matrix[0])
    for i in range(0,l):
        for j in range(i+1, l):
            all_links.append({
                'from': i,
                'to': j,
                'highDd': highD_dist_matrix[i][j],
                'graphd': graph_distances[i][j],
                'e': error_matrix[i][j]
            })
    return all_links

# create_list_of_all_links_with_values(
#         highD_dist_matrix,
#         mst
#         identify_error_in_distances(
#           highD_dist_matrix,
#           graph_to_distancematrix(mst)))
```

## Remove links that are already in MST

```python
all_links = create_list_of_all_links_with_values(highD_dist_matrix, mst, identify_error_in_distances(highD_dist_matrix, graph_to_distancematrix(mst)))
```

```python code_folding=[]
def remove_mst_links_from_list(list_of_links, graph):
    output = deepcopy(list_of_links)
    for e in graph.es():
        a = list(filter(lambda x:x['to'] == e.target, list_of_links))
        b = list(filter(lambda x:x['from'] == e.source, a))
        output.remove(b[0])
    return output
```

```python
all_links_without_mst = remove_mst_links_from_list(all_links, mst)
```

## Sort links

```python
sorted_links_to_add = deepcopy(all_links_without_mst)
sorted_links_to_add.sort(key = lambda x: x['highDd'])
```

# Add links to graph

```python code_folding=[]
def add_links_to_graph(list_of_links_to_add, n):
    new_graph = deepcopy(mst)
    new_graph.add_edges(list(map(lambda x:(x['from'],x['to']), sorted_links_to_add[:n])))
    return new_graph
```

```python
# g5 = add_links_to_graph(sorted_links_to_add, 5)
# draw_stad(g5)
```

# Simulated annealing
Taken from [https://perso.crans.org/besson/publis/notebooks/Simulated_annealing_in_Python.html](https://perso.crans.org/besson/publis/notebooks/Simulated_annealing_in_Python.html)

```python
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt  # to plot
import matplotlib as mpl

from scipy import optimize       # to compare

import seaborn as sns
sns.set(context="talk", style="darkgrid", palette="hls", font="sans-serif", font_scale=1.05)

FIGSIZE = (19, 8)  #: Figure size, in inches!
mpl.rcParams['figure.figsize'] = FIGSIZE
```

```python
interval = (0,200)

def clip(x):
    """ Force number of links added to be in the interval."""
    a, b = interval
    return max(min(x, b), a)
```

```python
def start():
    return 0
```

```python
def cost_function(nr_of_links):
    graph = add_links_to_graph(sorted_links_to_add, nr_of_links)
    return correlation_between_distance_matrices(highD_dist_matrix, list(graph_to_distancematrix(graph)))
```

```python
def random_neighbour(x, fraction=1):
    """Pick a new number of links, similar to what we already had."""
    amplitude = (max(interval) - min(interval)) * fraction / 10
    delta = int((-amplitude/2.) + amplitude * rn.random_sample())
    return clip(x + delta)
```

```python
def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        # print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        # print("    - Acceptance probabilty = {:.3g}...".format(p))
        return p
```

```python
def temperature(fraction):
    """ Example of temperature dicreasing as the process goes on."""
    return max(0.01, min(1, 1 - fraction))
```

```python
def see_annealing(states, costs):
    plt.figure()
    plt.suptitle("Evolution of states and costs of the simulated annealing")
    plt.subplot(121)
    plt.plot(states, 'lightblue')
    plt.title("States")
    plt.subplot(122)
    plt.plot(costs, 'lightgreen')
    plt.title("Costs")
    plt.show()
```

```python
def annealing(random_start,
              cost_function,
              random_neighbour,
              acceptance,
              temperature,
              maxsteps=1000,
              debug=True):
    """
    Optimize the black-box function 'cost_function' with the simulated annealing algorithm.
    IMPORTANT: the "state" is the number of links, NOT a graph
    """
    nr_of_links = random_start()
    cost = cost_function(nr_of_links)
    nrs_of_links, costs = [nr_of_links], [cost]
    for step in range(maxsteps):
        fraction = step / float(maxsteps)
        T = temperature(fraction)
        new_nr_of_links = random_neighbour(nr_of_links, fraction)
        if new_nr_of_links != nr_of_links:
            new_cost = cost_function(new_nr_of_links)
            if debug: print("Step #%d/%d:\tT=%.2f\tstate=%d\tnew_state=%d\tcost=%.2f\tnew_cost=%.2f"%(step, maxsteps, T, nr_of_links, new_nr_of_links, cost, new_cost))
            if acceptance_probability(cost, new_cost, T) > rn.random():
                nr_of_links, cost = new_nr_of_links, new_cost
                nrs_of_links.append(nr_of_links)
                costs.append(cost)
                if debug: print("\t\t\t\t\t\t\t\t==> Accepted")
            else:
                if debug: print("\t\t\t\t\t\t\t\t==> Rejected")
    return nr_of_links, cost_function(nr_of_links), nrs_of_links, costs
```

```python
annealing(start, cost_function, random_neighbour, acceptance_probability, temperature, maxsteps=30, debug=True);
```

```python
nr_of_links, cost, nrs_of_links, costs = annealing(start, cost_function, random_neighbour, acceptance_probability, temperature, maxsteps=8000, debug=False)
```

```python
see_annealing(nrs_of_links, costs)
```

```python

```
