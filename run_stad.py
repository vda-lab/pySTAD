import stad
import stad.visualize
import stad.graph

import click
import datetime
import numpy as np
import igraph as ig
import panel as pn

@click.command()
@click.argument('dataset', type=click.Choice(['circles', 'horse', 'simulated', 'barcelona'], case_sensitive=False))
@click.option('--nr_bins', default=5)
@click.option('--lens/--no-lens', 'use_lens', default=False)
@click.option('--corrected/--uncorrected', 'use_corrected', default=True)
def main(dataset, nr_bins, use_lens, use_corrected):
    start = datetime.datetime.now()
    original_highD_dist_matrix, lens_data, features = stad.load_testdata(dataset)

    if use_lens and len(lens_data) > 0:
        assigned_bins = stad.assign_bins(lens_data, nr_bins)
        highD_dist_matrix = stad.create_lensed_distmatrix_1step(original_highD_dist_matrix, assigned_bins)
    else:
        highD_dist_matrix = original_highD_dist_matrix

    mst = stad.graph.create_mst(highD_dist_matrix)
    one_sided_mst = np.where(stad.graph.triu_mask(mst, k=1), mst, 0)
    not_mst = stad.graph.masked_edges(highD_dist_matrix, mst == 0)
    edges, distances = stad.graph.ordered_edges(highD_dist_matrix, not_mst)  # We'll need distances later for STAD-R
    best_nr_of_links, best_correlation, xs, ys, dists, decision_colours, tmp_best_correlations, directions = stad.run_custom_basinhopping(one_sided_mst, not_mst, edges, distances, original_highD_dist_matrix, use_corrected)
    final_unit_adj = stad.create_final_unit_adj(one_sided_mst, edges, best_nr_of_links)
    final_weighted_adj = np.where(final_unit_adj == 1, original_highD_dist_matrix, 0)
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
    epochs = range(0, len(xs))
    for d in zip(xs,ys,decision_colours, tmp_best_correlations, directions, epochs):
        new_d = {}
        new_d['x'] = d[0]
        new_d['y'] = d[1]
        new_d['epoch'] = d[5]
        new_d['colour'] = d[2]
        new_d['tmp_best_correlation'] = d[3]
        new_d['more_links'] = d[4]
        my_d.append(new_d)

    pn.Column(
        pn.pane.Markdown("# QA plots for " + dataset),
        "Parameters:<br/>" +
        "  - lens: " + str(use_lens) + "<br/>" +
        "  - corrected: " + str(use_corrected) + "<br/>" +
        "Number of tested positions: " + str(len(xs)) + "<br/>" +
        "Number of links in mst: " + str(np.sum(one_sided_mst)) + "<br/>" +
        "Number of links added: " + str(best_x) + "<br/>" +
        "Number of links final: " + str(np.sum(final_unit_adj)) + "<br/>" +
        "Final correlation: <b>" + "{:.2f}".format(best_y) + "</b>",
        stad.visualize.plot_trend_vega(my_d),
        stad.visualize.plot_qa(one_sided_mst,not_mst,final_unit_adj,final_weighted_adj,highD_dist_matrix,best_x,best_y,distances,xs,ys),
        stad.visualize.draw_stad(g, features)
    ).show()


if __name__ == '__main__':
    main()
