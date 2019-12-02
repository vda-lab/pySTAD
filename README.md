# pySTAD - Python implementation of Simplified Topological Approximation of Data

## Example script
```python
# Load necessary libraries
import sTAD
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from IPython.display import Image, display
from Levenshtein import distance
import matplotlib as plt
from holoviews import opts

from bokeh.models import HoverTool
import matplotlib.pyplot as plt

import holoviews as hv
hv.extension('bokeh')

# Load data
data = pd.read_csv('data/five_circles.csv', header=0)
data['lens'] = data['hue']
values = data[['x','y']]
data = sTAD.assign_bins(data, 8)

# Calculate distance matrix
dist_matrix = euclidean_distances(values)

###################
# sTAD without lens
###################
mst_graph, mst, non_mst, cmdm, dm_distances = sTAD.create_mst(dist_matrix)
result = sTAD.find_stad_optimum(dist_matrix)
g = sTAD.create_network(result[0], mst, non_mst)

# Export to CSV files for gephi/tulip
import csv
g.vs['color'] = data['hue']
g.vs['bin'] = list(map(lambda x:str(x), (data['bin'])))
with open(dataset + '_' + str(result[0]) + '_nodes.csv', 'w') as f:
    file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['id','name','color','bin'])
    counter = 0
    for v in g.vs:
        file_writer.writerow([v.index, v.index, v['color'], v['bin']])
        counter += 1
with open(dataset + '_' + str(result[0]) + '_edges.csv', 'w') as f:
    file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['source','target'])
    for e in g.es:
        file_writer.writerow([e.tuple[0], e.tuple[1]])

################
# sTAD with lens
################
dist_matrix_with_lens = sTAD.alter_dist_matrix_1step(dist_matrix, data)
mst_graph_lens, mst_lens, non_mst_lens, cmdm_lens, dm_distances_lens = sTAD.create_mst(dist_matrix_with_lens)
result = sTAD.find_stad_optimum(dist_matrix_with_lens)
g = sTAD.create_network(result[0], mst_lens, non_mst_lens)

# Export to CSV files for gephi/tulip
import csv
g.vs['color'] = data['hue']
g.vs['bin'] = list(map(lambda x:str(x), (data['bin'])))
with open(dataset + '_' + str(result[0]) + '_lens_nodes.csv', 'w') as f:
    file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['id','name','color','bin'])
    counter = 0
    for v in g.vs:
        file_writer.writerow([v.index, v.index, v['color'], v['bin']])
        counter += 1
with open(dataset + '_' + str(result[0]) + '_lens_edges.csv', 'w') as f:
    file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['source','target'])
    for e in g.es:
        file_writer.writerow([e.tuple[0], e.tuple[1]])
```
