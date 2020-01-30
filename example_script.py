#!/usr/bin/env python
import stad
import pandas as pd

## Set `lens` to False to run the algorithm without lens

url = 'https://gist.githubusercontent.com/jandot/a84c0505cdc8008a6e5ae5032532a39f/raw/d834527117fd204d33486998d10290251354d013/five_circles.csv'
data = pd.read_csv(url, header=0)
values = data[['x','y']].values.tolist()
lens_values = data['hue'].map(lambda x:stad.hex_to_hsv(x)[0]).values

stad_graph = stad.run_stad(values, lens_values, lens=True)
stad.draw_stad('STAD graph of circle dataset', stad_graph)
