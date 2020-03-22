import stad.graph

import pandas as pd
import numpy as np
from scipy.sparse.csgraph import shortest_path
from math import exp
from numpy import log, log10
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


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

        x_min = min(sample[2])
        x_max = max(sample[2])
        lens = list(map(lambda x: normalise_number_between_0_and_255(x[2], x_min, x_max), sample))
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


def cost_function(nr_of_links, one_sided_mst, edges, distances, highD_dist_matrix):
    adj = stad.graph.add_unit_edges_to_matrix(one_sided_mst, edges[:nr_of_links])
    dist = shortest_path(adj, method="D", directed=False, unweighted=True)
    corr = np.corrcoef(dist.flatten(), highD_dist_matrix.flatten())[0][1]

    ### Add ratio
    nominator = np.sum(list(map(lambda x: 1-x, distances[:nr_of_links])))
    denominator = 1 + np.sum(distances[:nr_of_links])
    if denominator == 0:
        ratio = 1
    else:
        ratio = nominator / denominator

    corrected_corr = corr * ratio

    return corrected_corr, corr, dist


def take_step(previous_nr_of_links, max_links, temperature, more_links):
    stepsize = int(max_links * temperature)

    abs_random_jump_factor = np.abs(np.random.normal())
    if more_links:
        random_jump_factor = abs_random_jump_factor
    else:
        random_jump_factor = -abs_random_jump_factor

    nr_of_links = int(previous_nr_of_links + (random_jump_factor * stepsize))
    if nr_of_links < 0:
        nr_of_links = int(0.1*max_links) # if just 0, we might get stuck here
    elif nr_of_links > max_links:
        nr_of_links = int(0.9*max_links) # if just max_links, we might get stuck there

    return nr_of_links


def run_custom_basinhopping(one_sided_mst, not_mst, edges, distances, highD_dist_matrix, use_corrected):
    xs = []
    ys = []

    decision_colours = []
    tmp_best_results = []
    directions = [] # true if more, false if less
    dists = []
    max_links = int(np.sum(not_mst))
    best_result = 0
    best_nr_of_links = 0

    tmp_best_results.append(best_result)
    directions.append(True)

    temperatures = []
    nr_iterations = 100
    for i in range(1, nr_iterations):
        temperatures.append((log10(nr_iterations)-log10(i))/log10(nr_iterations))

    previous_nr_of_links = int(max_links/10)
    for temperature in temperatures:
        direction = directions[-1]
        nr_of_links = take_step(previous_nr_of_links, max_links, temperature, direction)
        previous_nr_of_links = nr_of_links

        new_corrected_corr, new_corr, dist = cost_function(nr_of_links, one_sided_mst, edges, distances, highD_dist_matrix)

        if use_corrected:
            new_result = new_corrected_corr
        else:
            new_result = new_corr

        if new_result > best_result:
            best_result = new_result
            best_nr_of_links = nr_of_links
            decision_colours.append('accepted')
            directions.append(directions[-1]) # if we're going in the right direction, keep going in that direction

        else:
            cutoff = exp((new_result-best_result)/temperature)
            random_number = np.random.random()
            if cutoff > random_number:
                directions.append(directions[-1])
                best_result = new_result
                best_nr_of_links = nr_of_links
                decision_colours.append('still-accepted')
            else:
                directions.append(not directions[-1])
                decision_colours.append('rejected')

        xs.append(nr_of_links)
        ys.append(new_corr)
        dists.append(dist)
        tmp_best_results.append(best_result)

    return best_nr_of_links, best_result, xs, ys, dists, decision_colours, tmp_best_results, directions


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
    return stad.graph.add_unit_edges_to_matrix(one_sided_mst, edges[:nr_of_links_to_add])


# RUN STAD
