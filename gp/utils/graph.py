import numpy as np
import torch
# import dgl

def construct_dgl_graph_from_edges(
    ori_head,
    ori_tail,
    n_entities,
    inverse_edge=False,
    edge_type=None,
    num_rels=None,
):
    num_rels = 1
    if inverse_edge:
        head = np.concatenate([ori_head, ori_tail])
        tail = np.concatenate([ori_tail, ori_head])
    else:
        head = ori_head
        tail = ori_tail
    g = dgl.graph((head, tail), num_nodes=n_entities)
    g.edata["src_node"] = torch.tensor(head, dtype=torch.long)
    g.edata["dst_node"] = torch.tensor(tail, dtype=torch.long)
    if edge_type is not None:
        if num_rels is None:
            num_rels = np.max(edge_type) + 1
        g.edata["type"] = torch.tensor(
            np.concatenate((edge_type, edge_type + num_rels))
        )
    return g


def sample_fixed_hop_size_neighbor(adj_mat, root, hop, max_nodes_per_hope=500):
    visited = np.array(root)
    fringe = np.array(root)
    nodes = np.array([])
    for h in range(1, hop + 1):
        u = adj_mat[fringe].nonzero()[1]
        fringe = np.setdiff1d(u, visited)
        visited = np.union1d(visited, fringe)
        if len(fringe) > max_nodes_per_hope:
            fringe = np.random.choice(fringe, max_nodes_per_hope)
        if len(fringe) == 0:
            break
        nodes = np.concatenate([nodes, fringe])
        # dist_list+=[dist+1]*len(fringe)
    return nodes


def get_k_hop_neighbors(adj_mat, root, hop, block_node=None):
    """Return k-hop neighbor dictionary of root.
    hop2neighbor[i] = the nodes that are exactly i distance away from root.
    """
    if block_node:
        visited = np.array([root, block_node])
    else:
        visited = np.array([root])
    fringe = np.array([root])
    hop2neighbor = {}
    hop2neighbor[0] = fringe
    for h in range(1, hop + 1):
        u = adj_mat[fringe].nonzero()[1]
        fringe = np.setdiff1d(u, visited)
        visited = np.union1d(visited, fringe)
        if len(fringe) == 0:
            break
        hop2neighbor[h] = fringe
        if block_node and h == 1:
            visited = np.setdiff1d(visited, np.array([block_node]))

    return hop2neighbor


def shortest_dist_sparse_mult(adj_mat, hop=6, source=None):
    if source is not None:
        neighbor_adj = adj_mat[source]
        ind = source
    else:
        neighbor_adj = adj_mat
        ind = np.arange(adj_mat.shape[0])
    neighbor_adj_set = [neighbor_adj]
    neighbor_dist = neighbor_adj.todense()
    for i in range(hop - 1):
        new_adj = neighbor_adj_set[i].dot(adj_mat)
        neighbor_adj_set.append(new_adj)
        update_ind = (new_adj.sign() - np.sign(neighbor_dist)) == 1
        r, c = update_ind.nonzero()
        neighbor_dist[r, c] = i + 2
    neighbor_dist[neighbor_dist < 1] = 9999
    neighbor_dist[np.arange(len(neighbor_dist)), ind] = 0
    return np.asarray(neighbor_dist)


def remove_gt_graph_edge(gt_graph, s, t):
    edges = gt_graph.edge(s, t, all_edges=True)
    for e in edges:
        gt_graph.remove_edge(e)
    if gt_graph.is_directed():
        edges = gt_graph.edge(t, s, all_edges=True)
        for e in edges:
            gt_graph.remove_edge(e)


def add_gt_graph_edge(gt_graph, s, t):
    gt_graph.add_edge(s, t)
    if gt_graph.is_directed():
        gt_graph.add_edge(t, s)