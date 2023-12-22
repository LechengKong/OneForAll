from typing import Union, Callable

from gp.utils.datasets import DatasetWithCollate
import torch_geometric as pyg
import torch
import numpy as np
from scipy.sparse import csr_array, coo_array
from gp.utils.graph import sample_fixed_hop_size_neighbor
from abc import ABC, abstractmethod
from utils import scipy_rwpe, set_mask

from gp.utils.utils import SmartTimer


class GraphTextDataset(DatasetWithCollate, ABC):
    """
    Base class for all OFA runtime datasets, responsible for loading graphs from OFAPygDataset, subgraphing,
    and prompt graph construction.
    """

    def __init__(self, graph: Union[pyg.data.Data, list[pyg.data.Data]], process_label_func: Callable, **kwargs):
        """
        Args:
            graph: Main graph objects, one single graph for single graph dataset, and list of graphs
                        for list of graphs.
            process_label_func: a Callable function that process the labels from original datasets to accommodate
                                    different tasks.
            **kwargs: additional arguments.
        """
        self.g = graph
        self.process_label_func = process_label_func
        self.kwargs = kwargs
        if "prompt_edge_list" in kwargs:
            self.prompt_edge_list = kwargs["prompt_edge_list"]
        else:
            self.prompt_edge_list = {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0], "c2n": [4, 0]},
        if "no_class_node" in kwargs and kwargs["no_class_node"]:
            self.no_class_node = True
        else:
            self.no_class_node = False

    def __getitem__(self, index):
        feature_graph = self.make_feature_graph(index)
        prompt_graph = self.make_prompted_graph(feature_graph)
        ret_data = self.to_pyg(feature_graph, prompt_graph)
        if "walk_length" in self.kwargs and self.kwargs["walk_length"] is not None:
            ret_data.rwpe = scipy_rwpe(ret_data, self.kwargs["walk_length"])
        return ret_data

    @abstractmethod
    def make_feature_graph(self, index) -> list:
        """
        Create feature subgraph based on index

        Args:
            index: int

        Returns:
            feat: torch.Tensor, node vector representations
            edge_feat: torch.Tensor, edge vector representations
            edge_index: torch.Tensor, feature edge indices
            e_type: torch.Tensor, feature edge types most likely 0-vector
            target_node_id: torch.Tensor, the indices of NOI
            class_emb: class node vector representations
            binary_rep: one-/multi-hot label representations
        """
        pass

    @abstractmethod
    def make_prompt_node(self, feat, class_emb):
        """
        Create prompt node features
        Args:
            feat: feature graph node features
            class_emb: class node features

        Returns:
            prompt_graph_node_features: prompt graph node features
        """
        pass

    def make_prompted_graph(self, feature_graph):
        """
        Create prompted graph based on feature graphs, prompt edge construction is based on self.prompt_edge_list.
        self.prompt_edge_list defines connection types, edge_type indices, and indices in to self.prompt_edge_emb.
        Refer to data.ofa_data.OFAPygDataset.get_edge_list for details.
        Args:
            feature_graph: output from self.make_feature_graph

        Returns:

        """
        (feat, edge_feat, edge_index, e_type, target_node_id, class_emb, label, binary_rep,) = feature_graph
        n_feat_node = len(feat)
        feat = self.make_prompt_node(feat, class_emb)
        prompt_edge_lst = []
        prompt_edge_type_lst = []
        prompt_edge_feat_lst = []
        for prompt_edge_str in self.prompt_edge_list:
            prompt_e_index = getattr(self, "make_" + prompt_edge_str + "_edge")(target_node_id, class_emb, n_feat_node)
            prompt_edge_types = torch.zeros(len(prompt_e_index[0]), dtype=torch.long) + \
                                self.prompt_edge_list[prompt_edge_str][0]
            if self.prompt_edge_list[prompt_edge_str][1] is None:
                edge_emb = self.prompt_edge_emb
            else:
                edge_emb = self.prompt_edge_emb[self.prompt_edge_list[prompt_edge_str][1]]
            prompt_edge_feat = edge_emb.repeat([len(prompt_e_index[0]), 1])
            prompt_edge_lst.append(prompt_e_index)
            prompt_edge_type_lst.append(prompt_edge_types)
            prompt_edge_feat_lst.append(prompt_edge_feat)
        edge_index = torch.cat([edge_index] + prompt_edge_lst, dim=-1, )
        e_type = torch.cat([e_type] + prompt_edge_type_lst)
        edge_feat = torch.cat([edge_feat] + prompt_edge_feat_lst)
        return feat, edge_index, label, edge_feat, e_type

    def to_pyg(self, feature_graph, prompted_graph):
        feat, edge_index, label, edge_feat, e_type = prompted_graph
        new_subg = pyg.data.Data(feat, edge_index, y=label, edge_attr=edge_feat, edge_type=e_type)
        num_class = len(feature_graph[-3])
        bin_labels = torch.zeros(new_subg.num_nodes, dtype=torch.float)
        bin_labels[new_subg.num_nodes - num_class:] = feature_graph[-1]
        new_subg.bin_labels = bin_labels
        set_mask(new_subg, "true_nodes_mask", list(range(new_subg.num_nodes - num_class, new_subg.num_nodes)))
        set_mask(new_subg, "noi_node_mask", new_subg.num_nodes - num_class - 1)
        set_mask(new_subg, "target_node_mask", feature_graph[-4])
        set_mask(new_subg, "feat_node_mask", list(range(len(feature_graph[0]))))
        new_subg.sample_num_nodes = new_subg.num_nodes
        new_subg.num_classes = num_class
        # print("text", new_subg)
        return new_subg

    def get_collate_fn(self):
        return pyg.loader.dataloader.Collater(None, None)

    def process_label(self, label):
        """
        Process labels into one-/multi-hot format using self.process_label_func
        """
        if self.process_label_func is None:
            trimed_class = torch.zeros((1, len(self.class_emb)))
            trimed_class[0, label] = 1
            return label, self.class_emb, trimed_class
        else:
            return self.process_label_func(self.class_emb, label)


class SubgraphDataset(GraphTextDataset):
    """
    Build feature subgraphs from a large graph, used mostly in node/link tasks
    """

    def __init__(self, pyg_graph, class_emb, prompt_edge_emb, data_idx, hop=2, class_mapping=None, to_undirected=False,
                 process_label_func=None, adj=None, **kwargs, ):
        super().__init__(pyg_graph, process_label_func, **kwargs)
        self.to_undirected = to_undirected
        edge_index = self.g.edge_index
        if self.to_undirected:
            edge_index = pyg.utils.to_undirected(edge_index)
        if adj is not None:
            self.adj = adj
        else:
            self.adj = csr_array((torch.ones(len(edge_index[0])), (edge_index[0], edge_index[1]),),
                                 shape=(self.g.num_nodes, self.g.num_nodes), )
        self.class_emb = class_emb
        self.prompt_edge_emb = prompt_edge_emb
        self.hop = hop
        self.data_idx = data_idx
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.data_idx)

    def get_neighbors(self, index):
        node_id = self.data_idx[index]
        neighbors = sample_fixed_hop_size_neighbor(self.adj, [node_id], self.hop, max_nodes_per_hope=100)
        neighbors = np.r_[node_id, neighbors]
        edges = self.adj[neighbors, :][:, neighbors].tocoo()
        if self.class_mapping is not None:
            label = self.class_mapping[self.g.y[node_id]]
        else:
            label = self.g.y[node_id]
        edge_index = torch.stack(
            [torch.tensor(edges.row, dtype=torch.long), torch.tensor(edges.col, dtype=torch.long), ])
        label, emb, binary_rep = self.process_label(label)
        return edge_index, neighbors, emb, label, binary_rep, [0]

    def make_feature_graph(self, index):
        (edge_index, neighbors, emb, label, binary_rep, target_node_id,) = self.get_neighbors(index)
        feat = self.g.node_text_feat[neighbors]
        e_type = torch.zeros(len(edge_index[0]), dtype=torch.long)
        edge_feat = self.g.edge_text_feat.repeat([len(edge_index[0]), 1])
        return (feat, edge_feat, edge_index, e_type, target_node_id, emb, label, binary_rep,)

    def make_prompt_node(self, feat, class_emb):
        # Only feature nodes and class nodes, no NOI node.
        if not self.no_class_node:
            feat = torch.cat([feat, class_emb], dim=0)
        return feat

    def make_f2n_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor(
            [target_node_id * len(class_emb), [i + n_feat_node for i in range(len(class_emb))], ], dtype=torch.long, )
        return prompt_edge

    def make_n2f_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor([[i + n_feat_node for i in range(len(class_emb))], target_node_id * len(class_emb)],
                                   dtype=torch.long, )
        return prompt_edge


class SubgraphNopromptDataset(SubgraphDataset):
    def make_prompted_graph(self, feature_graph):
        (feat, edge_feat, edge_index, e_type, target_node_id, label, binary_rep,) = feature_graph
        feat = torch.cat([feat, self.class_emb], dim=0)
        new_subg = pyg.data.Data(feat, edge_index, y=label, edge_attr=edge_feat, edge_type=e_type)
        return new_subg


class SubgraphHierDataset(SubgraphDataset):
    def __init__(self, pyg_graph, class_emb, prompt_edge_emb, noi_node_emb, data_idx, hop=2, class_mapping=None,
                 to_undirected=False, process_label_func=None, adj=None, **kwargs, ):
        super().__init__(pyg_graph, class_emb, prompt_edge_emb, data_idx, hop, class_mapping, to_undirected,
                         process_label_func, adj, **kwargs, )
        self.noi_node_emb = noi_node_emb

    def __len__(self):
        return len(self.data_idx)

    def make_prompt_node(self, feat, class_emb):
        # Add class node in zero-shot scenario. In few-shot scenario, only NOI node. Class nodes will be added by
        # future dataset wrapper
        if self.no_class_node:
            feat = torch.cat([feat, self.noi_node_emb], dim=0)
        else:
            feat = torch.cat([feat, self.noi_node_emb, class_emb], dim=0)

        return feat

    def make_f2n_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor([target_node_id, [n_feat_node] * len(target_node_id)], dtype=torch.long, )
        return prompt_edge

    def make_n2f_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor([[n_feat_node] * len(target_node_id), target_node_id], dtype=torch.long, )
        return prompt_edge

    def make_n2c_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor(
            [[n_feat_node] * len(class_emb), [i + n_feat_node + 1 for i in range(len(class_emb))], ],
            dtype=torch.long, )
        return prompt_edge

    def make_c2n_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor(
            [[i + n_feat_node + 1 for i in range(len(class_emb))], [n_feat_node] * len(class_emb)], dtype=torch.long, )
        return prompt_edge


class SubgraphLinkHierDataset(SubgraphHierDataset):
    def __init__(self, pyg_graph, class_emb, prompt_edge_emb, noi_node_emb, edges, remove_edge=False, hop=2,
                 class_mapping=None, to_undirected=False, process_label_func=None, adj=None, **kwargs, ):
        super().__init__(pyg_graph, class_emb, prompt_edge_emb, noi_node_emb, None, hop, class_mapping, to_undirected,
                         process_label_func, adj, **kwargs, )
        self.edges = edges
        self.pos_index = len(self.edges)
        self.remove_edge = remove_edge

        # Sample negative edges for training and testing
        dense_adj = self.adj.todense() == 0
        neg_row, neg_col = np.nonzero(dense_adj)
        neg_edge_idx = np.random.permutation(len(neg_row))[: self.pos_index]
        neg_row, neg_col = neg_row[neg_edge_idx], neg_col[neg_edge_idx]
        self.neg_edges = np.stack([neg_row, neg_col], axis=1)

        self.total_edges = np.concatenate([self.edges, self.neg_edges], axis=0)

    def __len__(self):
        return len(self.total_edges)

    def remove_link(self, row, col):
        remove_ind = np.logical_or(np.logical_and(row == 0, col == 1), np.logical_and(row == 1, col == 0), )
        keep_ind = np.logical_not(remove_ind)
        return row[keep_ind], col[keep_ind]

    def get_neighbors(self, index):
        edge_id = self.total_edges[index]

        if index < self.pos_index:
            label = 1
        else:
            label = 0
        node_ids = list(edge_id)
        neighbors = sample_fixed_hop_size_neighbor(self.adj, node_ids, self.hop, max_nodes_per_hope=100)
        neighbors = np.r_[node_ids, neighbors]
        edges = self.adj[neighbors, :][:, neighbors].tocoo()
        row = edges.row
        col = edges.col

        # Remove target edge from train graphs
        if self.remove_edge and index < self.pos_index:
            row, col = self.remove_link(row, col)
        edge_index = torch.stack([torch.tensor(row, dtype=torch.long), torch.tensor(col, dtype=torch.long), ])
        label, embs, binary_rep = self.process_label(label)
        return edge_index, neighbors, embs, label, binary_rep, [0, 1]


class SubgraphNopromptLinkDataset(SubgraphLinkHierDataset):
    def make_prompted_graph(self, feature_graph):
        (feat, edge_feat, edge_index, e_type, target_node_id, label, binary_rep,) = feature_graph
        feat = torch.cat([feat, self.class_emb], dim=0)
        new_subg = pyg.data.Data(feat, edge_index, y=label, edge_attr=edge_feat, edge_type=e_type)
        return new_subg


class SubgraphKGHierDataset(SubgraphHierDataset):
    def __init__(self, pyg_graph, class_emb, prompt_edge_emb, noi_node_emb, edges, remove_edge=False, hop=2,
                 class_mapping=None, to_undirected=False, process_label_func=None, adj=None, **kwargs, ):
        super().__init__(pyg_graph, class_emb, prompt_edge_emb, noi_node_emb, None, hop, class_mapping, to_undirected,
                         process_label_func, adj, **kwargs, )
        self.edges = edges
        self.remove_edge = remove_edge

    def __len__(self):
        return len(self.edges[0])

    def index_to_mask(self, index, size=None):
        size = int(index.max()) + 1 if size is None else size
        mask = torch.zeros(size, dtype=torch.bool)
        mask[index] = True
        return mask

    def remove_link(self, row, col, val, target_idx):
        keep_ind = val != target_idx
        return row[keep_ind], col[keep_ind], val[keep_ind]

    def get_neighbors(self, index):
        node_ids = list(self.edges[0][index])
        label = self.edges[1][index]

        neighbors = sample_fixed_hop_size_neighbor(self.adj, node_ids, self.hop, max_nodes_per_hope=100)
        neighbors = np.r_[node_ids, neighbors]
        node_mask = self.index_to_mask(neighbors, size=self.g.num_nodes)

        edge_mask = (node_mask[self.g.edge_index[0]] & node_mask[self.g.edge_index[1]])
        if self.remove_edge:
            index_mask = torch.ones(len(self.g.edge_index[0]), dtype=torch.bool)
            index_mask[index] = False
            edge_mask = edge_mask & index_mask
        edge2idx = torch.zeros(self.g.num_nodes, dtype=torch.long)
        edge2idx[neighbors] = torch.arange(len(neighbors))
        edge_index = self.g.edge_index[:, edge_mask]
        edge_type = self.g.edge_types[edge_mask]
        edge_index = edge2idx[edge_index]
        label, embs, binary_rep = self.process_label(label)
        return edge_index, neighbors, embs, label, binary_rep, [0, 1], edge_type

    def make_feature_graph(self, index):
        (edge_index, neighbors, embs, label, binary_rep, target_node_id, edge_type,) = self.get_neighbors(index)
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=-1)
        feat = self.g.node_text_feat[neighbors]
        e_type = torch.zeros(len(edge_index[0]), dtype=torch.long)

        # Inverse edge type index equals orignal edge type index plus # edge types.
        edge_feat = self.g.edge_text_feat[torch.cat([edge_type, edge_type + int(len(self.g.edge_text_feat) / 2)])]
        return (feat, edge_feat, edge_index, e_type, target_node_id, embs, label, binary_rep,)


class GraphListDataset(GraphTextDataset):
    """
    Dataset generate prompted graph from a list of graphs. Mostly used for graph tasks.
    """

    def __init__(self, graphs, class_embs, prompt_edge_emb, data_idx, process_label_func=None, **kwargs, ):
        super().__init__(graphs, process_label_func, **kwargs)
        self.class_emb = class_embs
        self.prompt_edge_emb = prompt_edge_emb
        self.data_idx = data_idx

    def __len__(self):
        return len(self.data_idx)

    def make_feature_graph(self, index):
        g = self.g[self.data_idx[index]]
        edge_index = g.edge_index
        label = g.y
        # label_emb = self.class_emb(label).view(1, -1)
        feat = g.node_text_feat
        edge_feat = g.edge_text_feat
        e_type = torch.zeros(len(edge_index[0]), dtype=torch.long)
        target_node_id = list(range(len(feat)))
        label, emb, binary_rep = self.process_label(label)
        return feat, edge_feat, edge_index, e_type, target_node_id, emb, label, binary_rep

    def make_prompt_node(self, feat, class_emb):
        if not self.no_class_node:
            feat = torch.cat([feat, class_emb], dim=0)
        return feat

    def make_f2n_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.stack([torch.arange(n_feat_node, dtype=torch.long).repeat(1, len(class_emb)).view(-1),
                                   torch.arange(n_feat_node, n_feat_node + len(class_emb),
                                                dtype=torch.long).repeat_interleave(n_feat_node), ], dim=0, )
        return prompt_edge

    def make_n2f_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.stack(
            [torch.arange(n_feat_node, n_feat_node + len(class_emb), dtype=torch.long).repeat_interleave(n_feat_node),
             torch.arange(n_feat_node, dtype=torch.long).repeat(1, len(class_emb)).view(-1)], dim=0, )
        return prompt_edge


class GraphListNopromptDataset(GraphListDataset):
    def make_prompted_graph(self, feature_graph):
        (feat, edge_feat, edge_index, next_nid, g_class_emb, label, trimmed_label,) = feature_graph
        feat = torch.cat([feat, g_class_emb], dim=0)
        edge_type = torch.zeros(len(edge_feat), dtype=torch.long)
        prompted_graph = pyg.data.Data(feat, edge_index, y=label, edge_attr=edge_feat, edge_type=edge_type, )
        # print(prompted_graph)
        # print(edge_index)
        return prompted_graph


class GraphListHierDataset(GraphListDataset):
    def __init__(self, graphs, class_embs, prompt_edge_emb, noi_node_emb, data_idx, process_label_func=None,
                 **kwargs, ):
        super().__init__(graphs, class_embs, prompt_edge_emb, data_idx, process_label_func, **kwargs, )
        self.noi_node_emb = noi_node_emb

    def make_prompt_node(self, feat, class_emb):
        if self.no_class_node:
            feat = torch.cat([feat, self.noi_node_emb], dim=0)
        else:
            feat = torch.cat([feat, self.noi_node_emb, class_emb], dim=0)
        return feat

    def make_f2n_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor([list(range(n_feat_node)), [n_feat_node] * n_feat_node], dtype=torch.long, )
        return prompt_edge

    def make_n2f_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor([[n_feat_node] * n_feat_node, list(range(n_feat_node))], dtype=torch.long, )
        return prompt_edge

    def make_n2c_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor(
            [[n_feat_node] * len(class_emb), [n_feat_node + i + 1 for i in range(len(class_emb))], ],
            dtype=torch.long, )
        return prompt_edge

    def make_c2n_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor(
            [[n_feat_node + i + 1 for i in range(len(class_emb))], [n_feat_node] * len(class_emb), ],
            dtype=torch.long, )
        return prompt_edge


class FewShotDataset(DatasetWithCollate):
    def __init__(self, fsmanager, query_graph_dataset, support_graph_dataset, fs_edge_feats, sample_size=1000):
        """
        FewShotDataset: use data indices generated from fsmanager to index into
        query_graph_dataset/support_graph_dataset (GraphTextDataset) to get query and support prompted graph only
        with NOI prompt node. It them assembles the graphs to a few-shot in-context prompted graphs.
        Args:
            fsmanager: query and support data indices manager
            query_graph_dataset: GraphTextDataset
            support_graph_dataset: GraphTextDataset
            fs_edge_feats: Few-shot edge features
            sample_size: number of samples, to be used with Dataloader
        """
        super().__init__()
        # mode 0 for sample index from training classes, 1 for val, 2 for test
        self.fs_idx_loader = fsmanager
        self.query_graph_dataset = query_graph_dataset
        self.support_graph_dataset = support_graph_dataset
        self.fs_edge_feats = fs_edge_feats
        self.sample_size = sample_size

    def get_noi_graph(self, dataset: GraphTextDataset, index, class_emb):
        feature_graph = list(dataset.make_feature_graph(index))
        feature_graph[-3] = class_emb
        prompted_graph = dataset.make_prompted_graph(feature_graph)
        return prompted_graph

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        # node_ids: (n_way, k_shot + 1)
        # node_cls: (n_way), representing true classes corresponding to n ways to query into class_emb
        node_ids, class_ind = self.fs_idx_loader.get_few_shot_idx()
        n_way = len(class_ind)
        k_shot = len(node_ids[0]) - 1
        q_query = 1
        class_emb = self.query_graph_dataset.class_emb[class_ind]

        # spt_subgraphs will store all n_way x k_shot subgraph info
        # qry subgraphs will store all n_way x q_query subgraph info
        qry_graphs, spt_graphs, final_subgraphs = [], [], []
        for cls_idx in range(len(node_ids)):
            for shot_idx in range(len(node_ids[0])):
                # sm.record()
                if shot_idx < q_query:
                    qry_graphs.append(
                        self.get_noi_graph(self.query_graph_dataset, node_ids[cls_idx, shot_idx], class_emb))
                else:
                    spt_graphs.append(
                        self.get_noi_graph(self.support_graph_dataset, node_ids[cls_idx, shot_idx], class_emb))

        # Randomly select one query node
        qry_ind = torch.randint(n_way, (1, 1))
        qry_graph = qry_graphs[qry_ind.view(-1)]
        graphs = [qry_graph] + spt_graphs
        feat_lst, edge_index, label, edge_feat, e_type = zip(*graphs)
        n_node = torch.tensor([len(feat) for feat in feat_lst])
        n_edge = torch.tensor([len(feat[0]) for feat in edge_index])
        noi_node_idx = torch.cumsum(n_node, dim=0)
        offset = torch.cat([torch.tensor([0]), noi_node_idx])[:-1]
        noi_node_idx = noi_node_idx - 1
        meta_feat = torch.cat(feat_lst, dim=0)
        meta_n_nodes = len(meta_feat)
        meta_feat = torch.cat([meta_feat, class_emb], dim=0)
        class_node_indices = torch.arange(meta_n_nodes, meta_n_nodes + n_way)
        spt_class_node_indices = class_node_indices.repeat_interleave(k_shot)
        meta_edge = torch.cat(edge_index, dim=-1) + offset.repeat_interleave(n_edge)
        qry_meta_edge = torch.stack([noi_node_idx[0].repeat(n_way), class_node_indices], dim=0)
        spt_meta_edge = torch.stack([noi_node_idx[1:], spt_class_node_indices], dim=0)
        meta_edge = torch.cat([meta_edge, qry_meta_edge, spt_meta_edge], dim=-1)

        meta_edge_feat = torch.cat(list(edge_feat) + [self.fs_edge_feats[0].repeat(len(qry_meta_edge[0]), 1),
                                                      self.fs_edge_feats[1].repeat(len(spt_meta_edge[0]), 1)], dim=0)
        meta_e_type = torch.cat(list(e_type) + [torch.zeros(len(qry_meta_edge[0]), dtype=torch.long) + 2,
                                                torch.zeros(len(spt_meta_edge[0]), dtype=torch.long) + 4])

        new_subg = pyg.data.Data(meta_feat, meta_edge, y=qry_ind, edge_attr=meta_edge_feat, edge_type=meta_e_type)

        bin_labels = torch.zeros(new_subg.num_nodes, dtype=torch.float)
        bin_labels[new_subg.num_nodes - n_way + qry_ind.view(-1):] = 1
        new_subg.bin_labels = bin_labels
        set_mask(new_subg, "true_nodes_mask", list(range(new_subg.num_nodes - n_way, new_subg.num_nodes)))
        set_mask(new_subg, "noi_node_mask", noi_node_idx)
        set_mask(new_subg, "target_node_mask", offset)
        set_mask(new_subg, "feat_node_mask", offset)
        new_subg.sample_num_nodes = new_subg.num_nodes
        new_subg.num_classes = n_way
        # print("text", new_subg)
        return new_subg

    def get_collate_fn(self):
        return pyg.loader.dataloader.Collater(None, None)


class MultiDataset(DatasetWithCollate):
    """
    One dataset that wraps different GraphTextDataset for training. It also dynamically manage the portion of
    the training datasets in each epoch based on validation results.
    """

    def __init__(self, datas, data_val_index=None, dataset_multiple=1, window_size=3, patience=3, min_ratio=0.1,
                 mode=None, ):
        self.datas = datas
        self.sizes = np.array([len(d) for d in datas])
        self.performance_record = []
        self.patience = patience
        self.data_val_index = data_val_index
        if self.data_val_index is None:
            self.data_val_index = [[i] for i in range(len(self.datas))]
        if isinstance(self.patience, int):
            self.patience = np.zeros(len(self.sizes)) + self.patience
        self.inpatience = np.zeros(len(self.patience))
        self.window_size = window_size
        if isinstance(self.window_size, int):
            self.window_size = np.zeros(len(self.sizes)) + self.window_size
        self.dataset_multiple = dataset_multiple
        if not isinstance(self.dataset_multiple, list):
            self.dataset_multiple = (np.zeros(len(self.sizes), dtype=float) + self.dataset_multiple)
        self.min_ratio = min_ratio
        if not isinstance(self.min_ratio, list):
            self.min_ratio = np.zeros(len(self.sizes), dtype=float) + self.min_ratio
        self.mode = mode
        if mode is not None:
            self.mode = np.array([1 if m == "max" else -1 for m in self.mode])
        # self.walk_length = walk_length
        self.compute_sizes()

    def compute_sizes(self):
        self.aug_sizes = (self.sizes * np.array(self.dataset_multiple)).astype(int)
        self.size_seg = np.cumsum(self.aug_sizes)
        self.ind2dataset = np.arange(len(self.datas)).repeat(self.aug_sizes)
        self.sample_ind = (np.random.rand(len(self.ind2dataset)) * self.sizes.repeat(self.aug_sizes)).astype(int)
        self.data_start_index = np.r_[0, self.size_seg[:-1]]

    def __len__(self):
        return np.sum(self.aug_sizes)

    def __getitem__(self, index):
        dataset_ind = self.ind2dataset[index]
        dataset = self.datas[dataset_ind]
        ret_data = dataset[self.sample_ind[index]]
        # if self.walk_length is not None:
        #     ret_data.rwpe = scipy_rwpe(ret_data, self.walk_length)
        #     print(ret_data.rwpe)
        return ret_data

    def get_collate_fn(self):
        return self.datas[0].get_collate_fn()

    def update(self, metric):
        metric = np.array(metric)
        p_records = np.array(self.performance_record)
        for i in range(len(self.datas)):
            if len(p_records) < self.window_size[i] or len(self.data_val_index[i]) == 0:
                continue

            vals = p_records[-int(self.window_size[i]):, self.data_val_index[i]]
            if self.mode is None:
                mode = np.ones(len(vals[0]), dtype=float)
            else:
                mode = self.mode[self.data_val_index[i]]
            mean = vals.mean()

            metric_vals = metric[self.data_val_index[i]]
            mean_improvement = (((metric_vals - mean) / mean) * mode).sum()
            if mean_improvement > 0:
                self.inpatience[i] = 0
            else:
                self.inpatience[i] += 1
            if self.inpatience[i] > self.patience[i]:
                self.dataset_multiple[i] = max(self.min_ratio[i],
                                               self.dataset_multiple[i] / 2)  # self.inpatience[i] = 0
        self.compute_sizes()
        self.performance_record.append(metric)
