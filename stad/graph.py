import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

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