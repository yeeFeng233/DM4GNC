import numpy as np
import scipy.sparse as sp
import torch


def sample_negative_edges_fast(adj, num_neg_samples, seed=None):
    """
    Fast negative edge sampling using set-based lookup
    
    Args:
        adj: sparse adjacency matrix
        num_neg_samples: number of negative edges to sample
        seed: random seed for reproducibility
    
    Returns:
        negative_edges: array of shape [num_neg_samples, 2]
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_nodes = adj.shape[0]
    
    # Convert existing edges to set for O(1) lookup
    edges_set = set()
    adj_coo = sp.coo_matrix(adj)
    for i, j in zip(adj_coo.row, adj_coo.col):
        edges_set.add((min(i, j), max(i, j)))  # Store as sorted tuple
    
    negative_edges = []
    max_attempts = num_neg_samples * 10  # Prevent infinite loop
    attempts = 0
    
    while len(negative_edges) < num_neg_samples and attempts < max_attempts:
        # Batch sample for efficiency
        batch_size = min(num_neg_samples - len(negative_edges), 1000)
        idx_i = np.random.randint(0, num_nodes, size=batch_size)
        idx_j = np.random.randint(0, num_nodes, size=batch_size)
        
        for i, j in zip(idx_i, idx_j):
            if i == j:
                continue
            edge = (min(i, j), max(i, j))
            if edge not in edges_set:
                negative_edges.append([i, j])
                edges_set.add(edge)  # Avoid duplicates in negative samples
                if len(negative_edges) >= num_neg_samples:
                    break
        
        attempts += batch_size
    
    return np.array(negative_edges)


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj,val_ratio=0.1):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def mask_test_edges_fast(adj, val_ratio=0.1, test_ratio=0.1, seed=None):
    """
    Fast version of mask_test_edges using set-based lookup
    
    Args:
        adj: sparse adjacency matrix
        val_ratio: validation edge ratio (default 0.05, i.e., 1/20)
        test_ratio: test edge ratio (default 0.1, i.e., 1/10)
        seed: random seed for reproducibility
    
    Returns:
        adj_train: training adjacency matrix
        train_edges, val_edges, val_edges_false, test_edges, test_edges_false
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    
    # Get upper triangular matrix (undirected graph)
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]  # All edges (upper triangular)
    
    num_edges = edges.shape[0]
    num_test = int(np.floor(num_edges * test_ratio))
    num_val = int(np.floor(num_edges * val_ratio))
    
    # Randomly split edges
    all_edge_idx = np.arange(num_edges)
    np.random.shuffle(all_edge_idx)
    
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    train_edge_idx = all_edge_idx[(num_val + num_test):]
    
    val_edges = edges[val_edge_idx]
    test_edges = edges[test_edge_idx]
    train_edges = edges[train_edge_idx]
    
    # Fast negative edge sampling using the new function
    print(f"Sampling {num_test} test negative edges...")
    test_edges_false = sample_negative_edges_fast(adj, num_test, seed=seed)
    
    print(f"Sampling {num_val} validation negative edges...")
    val_edges_false = sample_negative_edges_fast(adj, num_val, seed=seed+1 if seed else None)
    
    # Build training adjacency matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix(
        (data, (train_edges[:, 0], train_edges[:, 1])), 
        shape=adj.shape
    )
    adj_train = adj_train + adj_train.T  # Make symmetric
    
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def adj2edgeindex(adj):
    edge_index = []
    head, tail = torch.where(adj != 0)
    edge_index.append(head)
    edge_index.append(tail)
    return torch.stack(edge_index)