from gp.utils.datasets import DatasetWithCollate
import torch_geometric as pyg
import torch
import numpy as np
from scipy.sparse import csr_array, coo_array
from gp.utils.graph import sample_fixed_hop_size_neighbor
from fs_datamanager import FewShotDataManager
from torch_geometric.data import Batch
from torch_geometric.utils import dropout_edge
from utils import scipy_rwpe

from gp.utils.utils import SmartTimer


class GraphTextDataset(DatasetWithCollate):
    def __init__(self, graph, **kwargs):
        self.g = graph
        self.kwargs = kwargs

    def __getitem__(self, index):
        feature_graph = self.make_feature_graph(index)
        prompt_graph = self.make_prompted_graph(feature_graph)
        ret_data = self.to_pyg(feature_graph, prompt_graph)
        if (
            "walk_length" in self.kwargs
            and self.kwargs["walk_length"] is not None
        ):
            ret_data.rwpe = scipy_rwpe(ret_data, self.kwargs["walk_length"])
        return ret_data

    def make_feature_graph(self, index):
        pass

    def make_prompted_graph(self, feature_graph):
        pass

    def to_pyg(self, feature_graph, prompted_graph):
        pass

    def get_collate_fn(self):
        return pyg.loader.dataloader.Collater(None, None)


class SubgraphDataset(GraphTextDataset):
    def __init__(
        self,
        pyg_graph,
        class_emb,
        data_idx,
        hop=2,
        class_mapping=None,
        to_undirected=False,
        trim_class_func=None,
        adj=None,
        **kwargs,
    ):
        super().__init__(pyg_graph, **kwargs)
        self.to_undirected = to_undirected
        edge_index = self.g.edge_index
        if self.to_undirected:
            edge_index = pyg.utils.to_undirected(edge_index)
        if adj is not None:
            self.adj = adj
        else:
            self.adj = csr_array(
                (
                    torch.ones(len(edge_index[0])),
                    (edge_index[0], edge_index[1]),
                ),
                shape=(self.g.num_nodes, self.g.num_nodes),
            )
        self.class_emb = class_emb
        self.hop = hop
        self.data_idx = data_idx
        self.class_mapping = class_mapping
        self.trim_class_func = trim_class_func

    def __len__(self):
        return len(self.data_idx)

    def trim_class(self, label):
        if self.trim_class_func is None:
            trimed_class = torch.zeros((1, len(self.class_emb)))
            trimed_class[0, label] = 1
            return label, trimed_class
        else:
            return self.trim_class_func(label, len(self.class_emb))

    def get_neighbors(self, index):
        node_id = self.data_idx[index]
        neighbors = sample_fixed_hop_size_neighbor(
            self.adj, [node_id], self.hop, max_nodes_per_hope=100
        )
        neighbors = np.r_[node_id, neighbors]
        edges = self.adj[neighbors, :][:, neighbors].tocoo()
        if self.class_mapping is not None:
            label = self.class_mapping[self.g.y[node_id]]
        else:
            label = self.g.y[node_id]
        edge_index = torch.stack(
            [
                torch.tensor(edges.row, dtype=torch.long),
                torch.tensor(edges.col, dtype=torch.long),
            ]
        )
        label, binary_rep = self.trim_class(label)
        return edge_index, neighbors, label, binary_rep, [0]

    def make_feature_graph(self, index):
        (
            edge_index,
            neighbors,
            label,
            binary_rep,
            target_node_id,
        ) = self.get_neighbors(index)
        feat = self.g.x_text_feat[neighbors]
        e_type = torch.zeros(len(edge_index[0]), dtype=torch.long)
        edge_feat = self.g.edge_text_feat.repeat([len(edge_index[0]), 1])
        return (
            feat,
            edge_feat,
            edge_index,
            e_type,
            target_node_id,
            label,
            binary_rep,
        )

    def make_prompted_graph(self, feature_graph):
        (
            feat,
            edge_feat,
            edge_index,
            e_type,
            target_node_id,
            label,
            binary_rep,
        ) = feature_graph
        next_nid = len(feat)
        feat = torch.cat([feat, self.class_emb], dim=0)
        virtual_edge = torch.tensor(
            [
                target_node_id * len(self.class_emb),
                [i + next_nid for i in range(len(self.class_emb))],
            ],
            dtype=torch.long,
        )
        edge_index = torch.cat(
            [edge_index, virtual_edge, virtual_edge[[1, 0]]],
            dim=-1,
        )
        e_type = torch.cat(
            [
                e_type,
                torch.zeros(len(virtual_edge[0]), dtype=torch.long) + 1,
                torch.zeros(len(virtual_edge[0]), dtype=torch.long) + 2,
            ]
        )
        edge_feat = torch.cat(
            [
                edge_feat,
                self.g.prompt_edge_feat.repeat([len(virtual_edge[0]) * 2, 1]),
            ]
        )
        new_subg = pyg.data.Data(
            feat, edge_index, y=label, edge_attr=edge_feat, edge_type=e_type
        )
        return new_subg

    def to_pyg(self, feature_graph, prompted_graph):
        prompt_nodes_mask = torch.zeros(
            prompted_graph.num_nodes, dtype=torch.bool
        )
        bin_labels = torch.zeros(prompted_graph.num_nodes, dtype=torch.float)
        prompt_nodes_mask[
            prompted_graph.num_nodes - len(self.class_emb) :
        ] = True
        # noi_node_mask = torch.zeros(prompted_graph.num_nodes, dtype=torch.bool)
        # noi_node_mask[
        #     prompted_graph.num_nodes - len(self.class_emb) - 1
        # ] = True
        # prompted_graph.noi_node_mask = noi_node_mask
        prompted_graph.true_nodes_mask = prompt_nodes_mask
        bin_labels[
            prompted_graph.num_nodes - len(self.class_emb) :
        ] = feature_graph[-1]
        prompted_graph.bin_labels = bin_labels
        target_node_mask = torch.zeros(
            prompted_graph.num_nodes, dtype=torch.bool
        )
        target_node_mask[feature_graph[-3]] = True
        prompted_graph.target_node_mask = target_node_mask
        prompted_graph.sample_num_nodes = prompted_graph.num_nodes
        prompted_graph.num_classes = len(self.class_emb)
        # print("text", new_subg)
        return prompted_graph


class SubgraphNopromptDataset(SubgraphDataset):
    def make_prompted_graph(self, feature_graph):
        (
            feat,
            edge_feat,
            edge_index,
            e_type,
            target_node_id,
            label,
            binary_rep,
        ) = feature_graph
        feat = torch.cat([feat, self.class_emb], dim=0)
        new_subg = pyg.data.Data(
            feat, edge_index, y=label, edge_attr=edge_feat, edge_type=e_type
        )
        return new_subg


class SubgraphHierDataset(SubgraphDataset):
    def __init__(
        self,
        pyg_graph,
        class_emb,
        data_idx,
        hop=2,
        class_mapping=None,
        prompt_feat=None,
        to_undirected=False,
        trim_class_func=None,
        adj=None,
        **kwargs,
    ):
        super().__init__(
            pyg_graph,
            class_emb,
            data_idx,
            hop,
            class_mapping,
            to_undirected,
            trim_class_func,
            adj,
            **kwargs,
        )
        self.prompt_feat = prompt_feat

    def __len__(self):
        return len(self.data_idx)

    def make_prompted_graph(self, feature_graph):
        (
            feat,
            edge_feat,
            edge_index,
            e_type,
            target_node_id,
            label,
            binary_rep,
        ) = feature_graph
        next_nid = len(feat)
        feat = torch.cat([feat, self.prompt_feat, self.class_emb], dim=0)
        virtual_edge = torch.tensor(
            [
                target_node_id + [next_nid] * len(self.class_emb),
                [next_nid] * len(target_node_id)
                + [i + next_nid + 1 for i in range(len(self.class_emb))],
            ],
            dtype=torch.long,
        )
        edge_index = torch.cat(
            [edge_index, virtual_edge, virtual_edge[[1, 0]]],
            dim=-1,
        )
        e_type = torch.cat(
            [
                e_type,
                torch.zeros(1, dtype=torch.long) + 1,
                torch.zeros(int(len(virtual_edge[0]) - 1), dtype=torch.long)
                + 2,
                torch.zeros(1, dtype=torch.long) + 3,
                torch.zeros(int(len(virtual_edge[0]) - 1), dtype=torch.long)
                + 4,
            ]
        )
        edge_feat = torch.cat(
            [
                edge_feat,
                self.g.prompt_edge_feat.repeat([len(virtual_edge[0]) * 2, 1]),
            ]
        )
        # edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=-1)
        new_subg = pyg.data.Data(
            feat, edge_index, y=label, edge_attr=edge_feat, edge_type=e_type
        )
        return new_subg


class SubgraphLinkHierDataset(SubgraphHierDataset):
    def __init__(
        self,
        pyg_graph,
        class_emb,
        edges,
        remove_edge=False,
        hop=2,
        class_mapping=None,
        prompt_feat=None,
        to_undirected=False,
        trim_class_func=None,
        adj=None,
        **kwargs,
    ):
        super().__init__(
            pyg_graph,
            class_emb,
            None,
            hop,
            class_mapping,
            prompt_feat,
            to_undirected,
            trim_class_func,
            adj,
            **kwargs,
        )
        self.edges = edges
        self.pos_index = len(self.edges)
        self.remove_edge = remove_edge
        dense_adj = self.adj.todense() == 0
        neg_row, neg_col = np.nonzero(dense_adj)
        neg_edge_idx = np.random.permutation(len(neg_row))[: self.pos_index]
        neg_row, neg_col = neg_row[neg_edge_idx], neg_col[neg_edge_idx]
        self.neg_edges = np.stack([neg_row, neg_col], axis=1)

        self.total_edges = np.concatenate([self.edges, self.neg_edges], axis=0)

    def __len__(self):
        return len(self.total_edges)

    def remove_link(self, row, col):
        remove_ind = np.logical_or(
            np.logical_and(row == 0, col == 1),
            np.logical_and(row == 1, col == 0),
        )
        keep_ind = np.logical_not(remove_ind)
        return row[keep_ind], col[keep_ind]

    def get_neighbors(self, index):
        edge_id = self.total_edges[index]

        if index < self.pos_index:
            label = 1
        else:
            label = 0
        node_ids = list(edge_id)
        neighbors = sample_fixed_hop_size_neighbor(
            self.adj, node_ids, self.hop, max_nodes_per_hope=100
        )
        neighbors = np.r_[node_ids, neighbors]
        edges = self.adj[neighbors, :][:, neighbors].tocoo()
        row = edges.row
        col = edges.col
        if self.remove_edge and index < self.pos_index:
            row, col = self.remove_link(row, col)
        edge_index = torch.stack(
            [
                torch.tensor(row, dtype=torch.long),
                torch.tensor(col, dtype=torch.long),
            ]
        )
        label, binary_rep = self.trim_class(label)
        return edge_index, neighbors, label, binary_rep, [0, 1]


class SubgraphKGHierDataset(SubgraphHierDataset):
    def __init__(
        self,
        pyg_graph,
        class_emb,
        edges,
        remove_edge=False,
        hop=2,
        class_mapping=None,
        prompt_feat=None,
        to_undirected=False,
        trim_class_func=None,
        adj=None,
        **kwargs,
    ):
        super().__init__(
            pyg_graph,
            class_emb,
            None,
            hop,
            class_mapping,
            prompt_feat,
            to_undirected,
            trim_class_func,
            adj,
            **kwargs,
        )
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

        neighbors = sample_fixed_hop_size_neighbor(
            self.adj, node_ids, self.hop, max_nodes_per_hope=100
        )
        neighbors = np.r_[node_ids, neighbors]
        node_mask = self.index_to_mask(neighbors, size=self.g.num_nodes)

        edge_mask = (
            node_mask[self.g.edge_index[0]] & node_mask[self.g.edge_index[1]]
        )
        if self.remove_edge:
            index_mask = torch.ones(
                len(self.g.edge_index[0]), dtype=torch.bool
            )
            index_mask[index] = False
            edge_mask = edge_mask & index_mask
        edge2idx = torch.zeros(self.g.num_nodes, dtype=torch.long)
        edge2idx[neighbors] = torch.arange(len(neighbors))
        edge_index = self.g.edge_index[:, edge_mask]
        edge_type = self.g.edge_types[edge_mask]
        edge_index = edge2idx[edge_index]
        label, binary_rep = self.trim_class(label)
        return edge_index, neighbors, label, binary_rep, [0, 1], edge_type

    def make_feature_graph(self, index):
        (
            edge_index,
            neighbors,
            label,
            binary_rep,
            target_node_id,
            edge_type,
        ) = self.get_neighbors(index)
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=-1)
        feat = self.g.x_text_feat[neighbors]
        e_type = torch.zeros(len(edge_index[0]), dtype=torch.long)
        edge_feat = self.g.edge_text_feat[
            torch.cat(
                [edge_type, edge_type + int(len(self.g.edge_text_feat) / 2)]
            )
        ]
        return (
            feat,
            edge_feat,
            edge_index,
            e_type,
            target_node_id,
            label,
            binary_rep,
        )


class FewShotSubgraphDataset(SubgraphDataset):
    def __init__(
        self,
        pyg_graph,
        class_emb,
        data_idx,
        n_way: int,
        k_shot: int,
        q_query: int,
        datamanager: FewShotDataManager,
        mode: int,
        hop=2,
        class_mapping=None,
        prompt_feat=None,
        to_undirected=False,
        adj=None,
        single_prompt_edge=False,
        **kwargs,
    ):
        super().__init__(
            pyg_graph,
            class_emb,
            data_idx,
            hop,
            class_mapping,
            to_undirected,
            adj=adj,
            **kwargs,
        )
        # mode 0 for sample index from training classes, 1 for val, 2 for test
        self.fs_idx_loader = datamanager.get_data_loader(mode)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.prompt_feat = prompt_feat
        self.single_prompt_edge = single_prompt_edge

    def get_single_subgraph(self, node_id):
        """
        Get the info of node_id's khop subgraph
        Args:
            node_id: index of the node

        Returns:
            neighbors: feature matrix of all nodes in the subgraph
            edge_index: edge index tensor of the subgraph
            num_nodes: number of nodes in the subgraph
        """
        neighbors = sample_fixed_hop_size_neighbor(
            self.adj, [node_id], self.hop, max_nodes_per_hope=100
        )
        neighbors = np.r_[node_id, neighbors]
        edges = self.adj[neighbors, :][:, neighbors].tocoo()
        edge_index = torch.stack(
            [
                torch.tensor(edges.row, dtype=torch.long),
                torch.tensor(edges.col, dtype=torch.long),
            ]
        )
        return neighbors, edge_index, neighbors.shape[0]

    def make_feature_graph(self, graph_set):
        """
        For each query node, combine its subgraph with all support subgraphs.
        Args:
            qry_subgraph:
            spt_subgraphs:

        Returns:

        """
        (qry_subgraph, spt_subgraphs) = graph_set
        neighbors, edge_index, num_nodes = qry_subgraph
        neighbors_list = [neighbors] + [
            subgraph[0] for subgraph in spt_subgraphs
        ]
        feat = self.g.x_text_feat[np.concatenate(neighbors_list)]

        # nodes_pt represents the index of first node in each subgraph
        # nodes_pt[0] is the index of query node; nodes_pt[1,-1]: idx of support nodes; nodes_pt[-1]: idx of first prompt node
        nodes_pt = [0, num_nodes] + [subgraph[2] for subgraph in spt_subgraphs]
        nodes_pt = list(np.cumsum(nodes_pt))

        edge_index = torch.cat(
            [edge_index]
            + [
                spt_subgraphs[idx][1] + nodes_pt[idx + 1]
                for idx in range(len(spt_subgraphs))
            ],
            dim=1,
        )
        edge_feat = self.g.edge_text_feat.repeat([edge_index.size(1), 1])

        return feat, edge_index, edge_feat, nodes_pt

    def make_prompted_graph(self, feature_graph):
        (
            node_cls,
            feat,
            edge_index,
            edge_feat,
            nodes_pt,
            label,
            true_class,
        ) = feature_graph

        assert nodes_pt[-1] == (feat.shape[0])
        true_edge_num = len(edge_index[0])
        cls_feat = self.class_emb[node_cls]
        feat = torch.cat([feat, cls_feat])

        # Connect target node(index 0) with all class nodes
        new_node_id = 0
        qry_edge = torch.tensor(
            [
                [new_node_id] * self.n_way,
                [i + nodes_pt[-1] for i in range(self.n_way)],
            ],
            dtype=torch.long,
        )

        # Connect support nodes with corresponding class node
        spt_pt = nodes_pt[1:-1]
        cls_pt = [
            nodes_pt[-1] + i
            for i in range(self.n_way)
            for j in range(self.k_shot)
        ]
        spt_edge = torch.tensor(
            [spt_pt, cls_pt],
            dtype=torch.long,
        )

        # get final edge index and edge types
        edge_index = torch.cat(
            [
                edge_index,
                qry_edge,
                qry_edge[[1, 0]],
                spt_edge,
                spt_edge[[1, 0]],
            ],
            dim=-1,
        )
        e_type = torch.cat(
            [
                torch.zeros(true_edge_num, dtype=torch.long),
                torch.zeros(len(qry_edge[0]), dtype=torch.long) + 1,
                torch.zeros(len(qry_edge[0]), dtype=torch.long) + 2,
                torch.zeros(len(spt_edge[0]), dtype=torch.long) + 3,
                torch.zeros(len(spt_edge[0]), dtype=torch.long) + 4,
            ]
        )
        edge_feat = torch.cat(
            [
                edge_feat,
                self.g.prompt_edge_feat.repeat(
                    [(len(qry_edge[0]) + len(spt_edge[0])) * 2, 1]
                ),
            ]
        )
        assert edge_feat.size(0) == e_type.size(0)

        # get node masks
        new_subg = pyg.data.Data(
            feat, edge_index, y=label, edge_type=e_type, edge_attr=edge_feat
        )
        true_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        true_nodes_mask[-self.n_way :] = True
        target_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        target_node_mask[new_node_id] = True
        h_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        spt_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        spt_nodes_mask[nodes_pt[1:-1]] = True
        assert (spt_nodes_mask == True).sum() == self.n_way * self.k_shot

        new_subg.target_node_mask = target_node_mask
        new_subg.true_nodes_mask = true_nodes_mask
        new_subg.h_node_mask = h_node_mask
        new_subg.spt_nodes_mask = spt_nodes_mask
        new_subg.sample_num_nodes = new_subg.num_nodes
        new_subg.num_classes = self.n_way
        return new_subg

    def combine_final_graph(self, final_subgraphs):
        graph_feat = []
        graph_edge_index = []
        graph_edge_feat = []
        graph_y = []
        graph_e_type = []
        graph_target_node_mask = []
        graph_true_nodes_mask = []
        # graph_h_node_mask = []
        # graph_spt_nodes_mask = []
        pt = 0
        num_nodes = []

        for subgraph in final_subgraphs:
            graph_feat.append(subgraph.x)
            graph_edge_index.append(subgraph.edge_index + pt)
            graph_edge_feat.append(subgraph.edge_attr)
            # print(subgraph.edge_attr)
            graph_y.append(subgraph.y)
            graph_e_type.append(subgraph.edge_type)
            graph_target_node_mask.append(subgraph.target_node_mask)
            graph_true_nodes_mask.append(subgraph.true_nodes_mask)
            # graph_h_node_mask.append(subgraph.h_node_mask)
            # graph_spt_nodes_mask.append(subgraph.spt_nodes_mask)
            pt += subgraph.num_nodes
            num_nodes.append(subgraph.num_nodes)

        feat = torch.cat(graph_feat, dim=0)
        edge_index = torch.cat(graph_edge_index, dim=1)
        # print(graph_edge_feat)
        # print(len(graph_edge_feat))
        edge_feat = torch.cat(graph_edge_feat)
        y = torch.stack(graph_y, dim=0)
        e_type = torch.cat(graph_e_type, dim=0)
        target_node_mask = torch.cat(graph_target_node_mask, dim=0)
        true_nodes_mask = torch.cat(graph_true_nodes_mask, dim=0)
        # h_node_mask = torch.cat(graph_h_node_mask, dim=0)
        # spt_nodes_mask = torch.cat(graph_spt_nodes_mask, dim=0)

        new_subg = pyg.data.Data(
            feat, edge_index, y=y, edge_type=e_type, edge_attr=edge_feat
        )
        assert pt == new_subg.num_nodes
        new_subg.target_node_mask = target_node_mask
        new_subg.true_nodes_mask = true_nodes_mask
        # new_subg.h_node_mask = h_node_mask
        # new_subg.spt_nodes_mask = spt_nodes_mask
        new_subg.sample_num_nodes = new_subg.num_nodes
        # new_subg.num_classes = torch.full(
        #     (self.n_way * self.q_query,), self.n_way
        # )
        new_subg.num_classes = int(self.n_way)

        return new_subg

    def __getitem__(self, index):
        # sm = SmartTimer()
        # sm.record()
        # return node ids for an n_way k_shot q_query meta task
        # node_ids: (n_way, k_shot + q_query)
        # node_cls: (1, n_way), representing true classes corresponding to n ways
        node_ids, node_cls = next(iter(self.fs_idx_loader))

        max_n = self.n_way
        max_k = self.k_shot
        if self.kwargs["random_flag"]:
            if self.kwargs["min_n"] != self.n_way:
                self.n_way = torch.randint(
                    self.kwargs["min_n"], self.n_way, (1,)
                )[0]
            if self.kwargs["min_k"] > 0:
                if self.kwargs["min_k"] != self.k_shot:
                    self.k_shot = torch.randint(
                        self.kwargs["min_k"], self.k_shot, (1,)
                    )[0]
            else:
                self.k_shot = 0
        else:
            self.n_way = self.kwargs["min_n"]
            self.k_shot = self.kwargs["min_k"]
        node_cls = node_cls[: self.n_way]

        # spt_subgraphs will store all n_way x k_shot subgraph info
        # qry subgraphs will store all n_way x q_query subgraph info
        qry_subgraphs, spt_subgraphs, final_subgraphs = [], [], []
        for cls_idx in range(self.n_way):
            for shot_idx in range(self.k_shot + self.q_query):
                # sm.record()
                shot_subgraph = self.get_single_subgraph(
                    node_ids[cls_idx][shot_idx]
                )
                # sm.cal_and_update('get single subgraph')
                if shot_idx < self.q_query:
                    qry_subgraphs.append(shot_subgraph)
                else:
                    spt_subgraphs.append(shot_subgraph)
        assert len(qry_subgraphs) == (self.n_way * self.q_query)

        for idx, qry_subgraph in enumerate(qry_subgraphs):
            label = torch.tensor(idx // self.q_query)
            true_class = node_cls[label]
            # sm.record()
            feat, edge_index, edge_feat, nodes_pt = self.make_feature_graph(
                (qry_subgraph, spt_subgraphs)
            )
            # sm.cal_and_update('make feature graph')
            final_subgraph = self.make_prompted_graph(
                (
                    node_cls,
                    feat,
                    edge_index,
                    edge_feat,
                    nodes_pt,
                    label,
                    true_class,
                )
            )
            # sm.cal_and_update('make prompted graph')
            final_subgraphs.append(final_subgraph)

        meta_task_graph = self.combine_final_graph(final_subgraphs)
        bin_labels = torch.nn.functional.one_hot(
            meta_task_graph.y,
            self.n_way,
        ).flatten()
        total_labels = torch.zeros(meta_task_graph.num_nodes)
        total_labels[meta_task_graph.true_nodes_mask] = bin_labels.to(
            dtype=torch.float
        )
        meta_task_graph.bin_labels = total_labels
        meta_task_graph.y = torch.zeros((1, 1), dtype=torch.long)

        self.n_way = max_n
        self.k_shot = max_k

        return meta_task_graph


class FewShotHierDataset(FewShotSubgraphDataset):
    """
    FewShotHierDataset:
    true_nodes_mask: all n_way x q_query x n_way non-hierarchical prompt nodes
    target_node_mask: n_way x q_query query nodes
    spt_nodes_mask: n_way x q_query x n_way x k_shot support nodes
    h_node_mask: n_way x q_query hierarchical prompt nodes
    """

    def make_prompted_graph(self, feature_graph):
        (
            node_cls,
            feat,
            edge_index,
            edge_feat,
            nodes_pt,
            label,
            true_class,
        ) = feature_graph

        assert nodes_pt[-1] == (feat.shape[0])
        true_edge_num = len(edge_index[0])
        cls_feat = self.class_emb[node_cls]
        feat = torch.cat([feat, cls_feat, self.prompt_feat.view(1, -1)])

        new_node_id = 0

        # Connect support nodes with corresponding class node
        spt_pt = nodes_pt[1:-1]
        cls_pt = [
            nodes_pt[-1] + i
            for i in range(self.n_way)
            for j in range(self.k_shot)
        ]

        spt_edge = torch.tensor(
            [spt_pt, cls_pt],
            dtype=torch.long,
        )

        n_cls_pt = [nodes_pt[-1] + i for i in range(self.n_way)] + [
            nodes_pt[-1] + self.n_way
        ]
        h_pt = [nodes_pt[-1] + self.n_way for i in range(self.n_way)] + [
            new_node_id
        ]

        prompt_h_edge = torch.tensor(
            [
                h_pt,
                n_cls_pt,
            ],
            dtype=torch.long,
        )

        # get final edge index and edge types
        if self.single_prompt_edge:
            edge_index = torch.cat(
                [
                    edge_index,
                    spt_edge,
                    spt_edge[[1, 0]],
                    prompt_h_edge,
                ],
                dim=-1,
            )
            e_type = torch.cat(
                [
                    torch.zeros(true_edge_num, dtype=torch.long),
                    torch.zeros(len(spt_edge[0]), dtype=torch.long) + 1,
                    torch.zeros(len(spt_edge[0]), dtype=torch.long) + 2,
                    torch.zeros(len(prompt_h_edge[0]) - 1, dtype=torch.long)
                    + 3,
                    torch.zeros(1, dtype=torch.long) + 4,
                ]
            )
            edge_feat = torch.cat(
                [
                    edge_feat,
                    self.g.prompt_edge_feat.repeat(
                        [len(prompt_h_edge[0]) + len(spt_edge[0]) * 2, 1]
                    ),
                ]
            )
        else:
            edge_index = torch.cat(
                [
                    edge_index,
                    spt_edge,
                    spt_edge[[1, 0]],
                    prompt_h_edge,
                    prompt_h_edge[[1, 0]],
                ],
                dim=-1,
            )
            e_type = torch.cat(
                [
                    torch.zeros(true_edge_num, dtype=torch.long),
                    torch.zeros(len(spt_edge[0]), dtype=torch.long) + 1,
                    torch.zeros(len(spt_edge[0]), dtype=torch.long) + 2,
                    torch.zeros(len(prompt_h_edge[0]) - 1, dtype=torch.long)
                    + 3,
                    torch.zeros(1, dtype=torch.long) + 4,
                    torch.zeros(len(prompt_h_edge[0]) - 1, dtype=torch.long)
                    + 5,
                    torch.zeros(1, dtype=torch.long) + 6,
                ]
            )
            edge_feat = torch.cat(
                [
                    edge_feat,
                    self.g.prompt_edge_feat.repeat(
                        [(len(prompt_h_edge[0]) + len(spt_edge[0])) * 2, 1]
                    ),
                ]
            )
        assert edge_feat.size(0) == e_type.size(0)

        # get node masks
        new_subg = pyg.data.Data(
            feat, edge_index, y=label, edge_type=e_type, edge_attr=edge_feat
        )
        true_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        true_nodes_mask[nodes_pt[-1] : -1] = True
        assert (true_nodes_mask == True).sum() == self.n_way
        h_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        h_node_mask[-1] = True
        target_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        target_node_mask[new_node_id] = True
        spt_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        spt_nodes_mask[nodes_pt[1:-1]] = True
        assert (spt_nodes_mask == True).sum() == self.n_way * self.k_shot

        new_subg.target_node_mask = target_node_mask
        new_subg.h_node_mask = h_node_mask
        new_subg.true_nodes_mask = true_nodes_mask
        new_subg.spt_nodes_mask = spt_nodes_mask
        new_subg.sample_num_nodes = new_subg.num_nodes
        new_subg.num_classes = self.n_way
        return new_subg


class NewFewShotHierDataset(FewShotSubgraphDataset):
    def make_prompted_graph(self, feature_graph):
        (
            node_cls,
            feat,
            edge_index,
            edge_feat,
            nodes_pt,
            label,
            true_class,
        ) = feature_graph

        assert nodes_pt[-1] == (feat.shape[0])
        true_edge_num = len(edge_index[0])
        cls_feat = self.class_emb[node_cls]
        feat = torch.cat(
            [
                feat,
                cls_feat,
                self.prompt_feat.view(1, -1).repeat(1 + self.n_way, 1),
            ]
        )

        new_node_id = 0

        # Connect support nodes with corresponding class node
        spt_pt = nodes_pt[1:-1]
        cls_pt = [
            nodes_pt[-1] + i + 1 + self.n_way
            for i in range(self.n_way)
            for j in range(self.k_shot)
        ]

        spt_edge = torch.tensor(
            [spt_pt, cls_pt],
            dtype=torch.long,
        )

        # Connect spt_h_nodes with h_nodes
        spt_h_nodes = [
            nodes_pt[-1] + 1 + self.n_way + i for i in range(self.n_way)
        ]
        prompt_nodes = [nodes_pt[-1] + i for i in range(self.n_way)]
        spt_prompt_edge = torch.tensor(
            [
                spt_h_nodes,
                prompt_nodes,
            ],
            dtype=torch.long,
        )

        qry_h_edge = torch.tensor(
            [
                [nodes_pt[-1] + self.n_way for i in range(self.n_way)]
                + [new_node_id],
                [nodes_pt[-1] + i for i in range(self.n_way)]
                + [nodes_pt[-1] + self.n_way],
            ],
            dtype=torch.long,
        )

        # get final edge index and edge types
        if self.single_prompt_edge:
            edge_index = torch.cat(
                [
                    edge_index,
                    spt_edge,
                    spt_prompt_edge,
                    qry_h_edge,
                ],
                dim=-1,
            )
            e_type = torch.cat(
                [
                    torch.zeros(true_edge_num, dtype=torch.long),
                    torch.zeros(len(spt_edge[0]), dtype=torch.long) + 1,
                    torch.zeros(len(spt_prompt_edge[0]), dtype=torch.long) + 2,
                    torch.zeros(len(qry_h_edge[0]) - 1, dtype=torch.long) + 3,
                    torch.zeros(1, dtype=torch.long) + 4,
                ]
            )
            edge_feat = torch.cat(
                [
                    edge_feat,
                    self.g.prompt_edge_feat.repeat(
                        [
                            len(spt_prompt_edge[0])
                            + len(spt_edge[0])
                            + len(qry_h_edge[0]),
                            1,
                        ]
                    ),
                ]
            )
        assert edge_feat.size(0) == e_type.size(0)

        # get node masks
        new_subg = pyg.data.Data(
            feat, edge_index, y=label, edge_type=e_type, edge_attr=edge_feat
        )
        true_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        true_nodes_mask[nodes_pt[-1] : nodes_pt[-1] + self.n_way] = True
        assert (true_nodes_mask == True).sum() == self.n_way
        h_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        h_node_mask[nodes_pt[-1] + self.n_way :] = True
        target_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        target_node_mask[new_node_id] = True
        spt_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        spt_nodes_mask[nodes_pt[1:-1]] = True
        assert (spt_nodes_mask == True).sum() == self.n_way * self.k_shot

        new_subg.target_node_mask = target_node_mask
        new_subg.h_node_mask = h_node_mask
        new_subg.true_nodes_mask = true_nodes_mask
        new_subg.spt_nodes_mask = spt_nodes_mask
        new_subg.sample_num_nodes = new_subg.num_nodes
        new_subg.num_classes = self.n_way
        return new_subg


class FewShotNCDataset(FewShotSubgraphDataset):
    def make_prompted_graph(self, feature_graph):
        (
            node_cls,
            feat,
            edge_index,
            edge_feat,
            nodes_pt,
            label,
            true_class,
        ) = feature_graph

        assert nodes_pt[-1] == (feat.shape[0])
        true_edge_num = len(edge_index[0])
        cls_feat = self.class_emb[node_cls]
        feat = torch.cat(
            [
                feat,
                cls_feat,
                self.prompt_feat.view(1, -1).repeat(
                    1 + self.n_way * self.k_shot, 1
                ),
            ]
        )

        new_node_id = 0

        # Connect support nodes with corresponding class node
        spt_pt = nodes_pt[1:-1]
        spt_h_nodes = [
            nodes_pt[-1] + self.n_way + 1 + i
            for i in range(self.n_way * self.k_shot)
        ]

        spt_edge = torch.tensor(
            [
                spt_pt,
                spt_h_nodes,
            ],
            dtype=torch.long,
        )

        # Connect spt_h_nodes with h_nodes
        prompt_nodes = [
            nodes_pt[-1] + i
            for i in range(self.n_way)
            for j in range(self.k_shot)
        ]
        spt_prompt_edge = torch.tensor(
            [
                spt_h_nodes,
                prompt_nodes,
            ],
            dtype=torch.long,
        )

        # Connect qry_h_node with h_nodes, Connect qry node with qry_h_node
        qry_h_edge = torch.tensor(
            [
                [nodes_pt[-1] + self.n_way for i in range(self.n_way)]
                + [new_node_id],
                [nodes_pt[-1] + i for i in range(self.n_way)]
                + [nodes_pt[-1] + self.n_way],
            ],
            dtype=torch.long,
        )

        # get final edge index and edge types
        if self.single_prompt_edge:
            edge_index = torch.cat(
                [
                    edge_index,
                    spt_edge,
                    spt_prompt_edge,
                    qry_h_edge,
                ],
                dim=-1,
            )
            e_type = torch.cat(
                [
                    torch.zeros(true_edge_num, dtype=torch.long),
                    torch.zeros(len(spt_edge[0]), dtype=torch.long) + 1,
                    torch.zeros(len(spt_prompt_edge[0]), dtype=torch.long) + 4,
                    torch.zeros(len(qry_h_edge[0]) - 1, dtype=torch.long) + 3,
                    torch.zeros(1, dtype=torch.long) + 1,
                ]
            )
            edge_feat = torch.cat(
                [
                    edge_feat,
                    self.g.prompt_edge_feat.repeat(
                        [
                            len(spt_prompt_edge[0])
                            + len(spt_edge[0])
                            + len(qry_h_edge[0]),
                            1,
                        ]
                    ),
                ]
            )
        assert edge_feat.size(0) == e_type.size(0)

        # get node masks
        new_subg = pyg.data.Data(
            feat, edge_index, y=label, edge_type=e_type, edge_attr=edge_feat
        )
        true_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        true_nodes_mask[nodes_pt[-1] : nodes_pt[-1] + self.n_way] = True
        assert (true_nodes_mask == True).sum() == self.n_way
        h_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        h_node_mask[nodes_pt[-1] + self.n_way :] = True
        target_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        target_node_mask[new_node_id] = True
        spt_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        spt_nodes_mask[nodes_pt[1:-1]] = True
        assert (spt_nodes_mask == True).sum() == self.n_way * self.k_shot

        new_subg.target_node_mask = target_node_mask
        new_subg.h_node_mask = h_node_mask
        new_subg.true_nodes_mask = true_nodes_mask
        new_subg.spt_nodes_mask = spt_nodes_mask
        new_subg.sample_num_nodes = new_subg.num_nodes
        new_subg.num_classes = self.n_way
        return new_subg


class CLFewShotHierDataset(NewFewShotHierDataset):
    def __init__(
        self,
        pyg_graph,
        class_emb,
        data_idx,
        n_way: int,
        k_shot: int,
        q_query: int,
        drop_feature_ratio1,
        drop_feature_ratio2,
        drop_edge_ratio1,
        drop_edge_ratio2,
        datamanager: FewShotDataManager,
        mode: int,
        hop=2,
        class_mapping=None,
        prompt_feat=None,
        to_undirected=False,
        adj=None,
        single_prompt_edge=False,
        **kwargs,
    ):
        super().__init__(
            pyg_graph,
            class_emb,
            data_idx,
            n_way,
            k_shot,
            q_query,
            datamanager,
            mode,
            hop,
            class_mapping,
            prompt_feat,
            to_undirected,
            adj,
            single_prompt_edge,
            **kwargs,
        )
        self.df = [drop_feature_ratio1, drop_feature_ratio2]
        self.de = [drop_edge_ratio1, drop_edge_ratio2]

    def drop_feature(self, x: torch.Tensor, drop_prob: float) -> torch.Tensor:
        device = x.device
        drop_mask = (
            torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1)
            < drop_prob
        )
        drop_mask = drop_mask.to(device)
        x = x.clone()
        x[:, drop_mask] = 0
        return x

    def get_aug_graph(self, node_ids, node_cls, aug_param_idx):
        qry_subgraphs, spt_subgraphs, final_subgraphs = [], [], []
        for cls_idx in range(self.n_way):
            for shot_idx in range(self.k_shot + self.q_query):
                shot_subgraph = self.get_single_subgraph(
                    node_ids[cls_idx][shot_idx]
                )
                if shot_idx < self.q_query:
                    qry_subgraphs.append(shot_subgraph)
                else:
                    spt_subgraphs.append(shot_subgraph)
        assert len(qry_subgraphs) == (self.n_way * self.q_query)
        assert len(spt_subgraphs) == (self.n_way * self.k_shot)

        for idx, qry_subgraph in enumerate(qry_subgraphs):
            label = torch.tensor(idx // self.q_query)
            true_class = node_cls[label]
            feat, edge_index, edge_feat, nodes_pt = self.make_feature_graph(
                (qry_subgraph, spt_subgraphs)
            )
            # drop edge
            edge_index, edge_mask = dropout_edge(
                edge_index, p=self.de[aug_param_idx], force_undirected=True
            )
            edge_feat = edge_feat[edge_mask]

            final_subgraph = self.make_prompted_graph(
                (
                    node_cls,
                    feat,
                    edge_index,
                    edge_feat,
                    nodes_pt,
                    label,
                    true_class,
                )
            )
            final_subgraphs.append(final_subgraph)

        return final_subgraphs

    def __getitem__(self, index):
        # return node ids for an n_way k_shot q_query meta task
        # node_ids: (n_way, k_shot + q_query)
        # node_cls: (1, n_way), representing true classes corresponding to n ways
        node_ids, node_cls = next(iter(self.fs_idx_loader))

        a1 = self.get_aug_graph(node_ids, node_cls, 0)
        a2 = self.get_aug_graph(node_ids, node_cls, 1)

        meta_task_graph1 = self.combine_final_graph(a1)
        # drop feature
        meta_task_graph1.x = self.drop_feature(meta_task_graph1.x, self.df[0])
        meta_task_graph2 = self.combine_final_graph(a2)
        # drop feature
        meta_task_graph2.x = self.drop_feature(meta_task_graph2.x, self.df[0])
        bin_labels = torch.nn.functional.one_hot(
            meta_task_graph1.y,
            self.n_way,
        ).flatten()
        meta_task_graph1.bin_labels = bin_labels.to(dtype=torch.float)
        bin_labels = torch.nn.functional.one_hot(
            meta_task_graph2.y,
            self.n_way,
        ).flatten()
        meta_task_graph2.bin_labels = bin_labels.to(dtype=torch.float)

        return meta_task_graph1, meta_task_graph2

    def get_collate_fn(self):
        def aug_collate(batch):
            # Separating the subgraphs
            subgraph1_list, subgraph2_list = list(zip(*batch))

            # Creating two separate batches
            batch1 = Batch.from_data_list(subgraph1_list)
            batch2 = Batch.from_data_list(subgraph2_list)

            return batch1, batch2

        return aug_collate


class ZeroShotNCDataset(FewShotSubgraphDataset):
    def make_feature_graph(self, graph_set):
        (qry_subgraph, _) = graph_set
        neighbors, edge_index, num_nodes = qry_subgraph
        feat = self.g.x_text_feat[neighbors.astype(int)]
        nodes_pt = [0, num_nodes]
        edge_feat = self.g.edge_text_feat.repeat([edge_index.size(1), 1])
        return feat, edge_index, edge_feat, nodes_pt

    def make_prompted_graph(self, feature_graph):
        (
            node_cls,
            feat,
            edge_index,
            edge_feat,
            nodes_pt,
            label,
            true_class,
        ) = feature_graph

        assert nodes_pt[-1] == (feat.shape[0])
        true_edge_num = len(edge_index[0])
        cls_feat = self.class_emb[node_cls]
        feat = torch.cat([feat, cls_feat, self.prompt_feat.view(1, -1)])

        new_node_id = 0

        # TODO: one direction edge. qry to hnode, hnode to prompt nodes
        h_pt = [nodes_pt[-1] + self.n_way for i in range(self.n_way)] + [
            new_node_id
        ]
        n_cls_pt = [nodes_pt[-1] + i for i in range(self.n_way)] + [
            nodes_pt[-1] + self.n_way
        ]

        prompt_h_edge = torch.tensor(
            [
                h_pt,
                n_cls_pt,
            ],
            dtype=torch.long,
        )

        # get final edge index and edge types
        edge_index = torch.cat(
            [edge_index, prompt_h_edge],
            dim=-1,
        )

        e_type = torch.cat(
            [
                torch.zeros(true_edge_num, dtype=torch.long),
                torch.zeros(len(prompt_h_edge[0]) - 1, dtype=torch.long) + 3,
                torch.zeros(1, dtype=torch.long) + 1,
            ]
        )

        edge_feat = torch.cat(
            [
                edge_feat,
                self.g.prompt_edge_feat.repeat([len(prompt_h_edge[0]), 1]),
            ]
        )

        assert edge_feat.size(0) == e_type.size(0)

        # get node masks
        new_subg = pyg.data.Data(
            feat, edge_index, y=label, edge_type=e_type, edge_attr=edge_feat
        )
        true_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        true_nodes_mask[nodes_pt[-1] : -1] = True
        assert (true_nodes_mask == True).sum() == self.n_way
        h_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        h_node_mask[-1] = True
        target_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        target_node_mask[new_node_id] = True
        spt_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        assert (spt_nodes_mask == True).sum() == self.n_way * self.k_shot

        new_subg.target_node_mask = target_node_mask
        new_subg.h_node_mask = h_node_mask
        new_subg.true_nodes_mask = true_nodes_mask
        new_subg.spt_nodes_mask = spt_nodes_mask
        new_subg.sample_num_nodes = new_subg.num_nodes
        new_subg.num_classes = self.n_way
        return new_subg


class ZeroShotSubgraphDataset(ZeroShotNCDataset):
    def make_prompted_graph(self, feature_graph):
        (
            node_cls,
            feat,
            edge_index,
            edge_feat,
            nodes_pt,
            label,
            true_class,
        ) = feature_graph

        assert nodes_pt[-1] == (feat.shape[0])
        true_edge_num = len(edge_index[0])
        cls_feat = self.class_emb[node_cls]
        feat = torch.cat([feat, cls_feat])

        new_node_id = 0

        # Add edge between all prompt nodes and query node
        n_cls_pt = [nodes_pt[-1] + i for i in range(self.n_way)]
        qry_pt = [new_node_id] * self.n_way
        prompt_edge = torch.tensor(
            [
                n_cls_pt,
                qry_pt,
            ],
            dtype=torch.long,
        )

        # get final edge index and edge types
        edge_index = torch.cat(
            [edge_index, prompt_edge, prompt_edge[[1, 0]]],
            dim=-1,
        )

        e_type = torch.cat(
            [
                torch.zeros(true_edge_num, dtype=torch.long),
                torch.zeros(len(prompt_edge[0]), dtype=torch.long) + 1,
                torch.zeros(len(prompt_edge[0]), dtype=torch.long) + 2,
            ]
        )

        edge_feat = torch.cat(
            [
                edge_feat,
                self.g.prompt_edge_feat.repeat([len(prompt_edge[0]) * 2, 1]),
            ]
        )

        assert edge_feat.size(0) == e_type.size(0)

        # get node masks
        new_subg = pyg.data.Data(
            feat, edge_index, y=label, edge_type=e_type, edge_attr=edge_feat
        )
        true_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        true_nodes_mask[nodes_pt[-1] :] = True
        assert (true_nodes_mask == True).sum() == self.n_way
        h_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        target_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        target_node_mask[new_node_id] = True
        spt_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        assert (spt_nodes_mask == True).sum() == self.n_way * self.k_shot

        new_subg.target_node_mask = target_node_mask
        new_subg.h_node_mask = h_node_mask
        new_subg.true_nodes_mask = true_nodes_mask
        new_subg.spt_nodes_mask = spt_nodes_mask
        new_subg.sample_num_nodes = new_subg.num_nodes
        new_subg.num_classes = self.n_way
        return new_subg


class CLZeroShotNCDataset(ZeroShotNCDataset):
    def __init__(
        self,
        pyg_graph,
        class_emb,
        data_idx,
        n_way: int,
        k_shot: int,
        q_query: int,
        drop_feature_ratio1,
        drop_feature_ratio2,
        drop_edge_ratio1,
        drop_edge_ratio2,
        datamanager: FewShotDataManager,
        mode: int,
        hop=2,
        class_mapping=None,
        prompt_feat=None,
        to_undirected=False,
        adj=None,
        **kwargs,
    ):
        super().__init__(
            pyg_graph,
            class_emb,
            data_idx,
            n_way,
            k_shot,
            q_query,
            datamanager,
            mode,
            hop,
            class_mapping,
            prompt_feat,
            to_undirected,
            adj,
            **kwargs,
        )
        self.df = [drop_feature_ratio1, drop_feature_ratio2]
        self.de = [drop_edge_ratio1, drop_edge_ratio2]

    def drop_feature(self, x: torch.Tensor, drop_prob: float) -> torch.Tensor:
        device = x.device
        drop_mask = (
            torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1)
            < drop_prob
        )
        drop_mask = drop_mask.to(device)
        x = x.clone()
        x[:, drop_mask] = 0
        return x

    def get_aug_graph(self, node_ids, node_cls, aug_param_idx):
        qry_subgraphs, spt_subgraphs, final_subgraphs = [], [], []
        for cls_idx in range(self.n_way):
            for shot_idx in range(self.k_shot + self.q_query):
                shot_subgraph = self.get_single_subgraph(
                    node_ids[cls_idx][shot_idx]
                )
                if shot_idx < self.q_query:
                    qry_subgraphs.append(shot_subgraph)
                else:
                    spt_subgraphs.append(shot_subgraph)
        assert len(qry_subgraphs) == (self.n_way * self.q_query)
        assert len(spt_subgraphs) == (self.n_way * self.k_shot)

        for idx, qry_subgraph in enumerate(qry_subgraphs):
            label = torch.tensor(idx // self.q_query)
            true_class = node_cls[label]
            feat, edge_index, edge_feat, nodes_pt = self.make_feature_graph(
                (qry_subgraph, spt_subgraphs)
            )
            # drop edge
            edge_index, edge_mask = dropout_edge(
                edge_index, p=self.de[aug_param_idx], force_undirected=True
            )
            edge_feat = edge_feat[edge_mask]

            final_subgraph = self.make_prompted_graph(
                (
                    node_cls,
                    feat,
                    edge_index,
                    edge_feat,
                    nodes_pt,
                    label,
                    true_class,
                )
            )
            final_subgraphs.append(final_subgraph)

        return final_subgraphs

    def __getitem__(self, index):
        # return node ids for an n_way k_shot q_query meta task
        # node_ids: (n_way, k_shot + q_query)
        # node_cls: (1, n_way), representing true classes corresponding to n ways
        node_ids, node_cls = next(iter(self.fs_idx_loader))

        a1 = self.get_aug_graph(node_ids, node_cls, 0)
        a2 = self.get_aug_graph(node_ids, node_cls, 1)

        meta_task_graph1 = self.combine_final_graph(a1)
        # drop feature
        meta_task_graph1.x = self.drop_feature(meta_task_graph1.x, self.df[0])
        meta_task_graph2 = self.combine_final_graph(a2)
        # drop feature
        meta_task_graph2.x = self.drop_feature(meta_task_graph2.x, self.df[0])
        bin_labels = torch.nn.functional.one_hot(
            meta_task_graph1.y,
            self.n_way,
        ).flatten()
        meta_task_graph1.bin_labels = bin_labels.to(dtype=torch.float)
        bin_labels = torch.nn.functional.one_hot(
            meta_task_graph2.y,
            self.n_way,
        ).flatten()
        meta_task_graph2.bin_labels = bin_labels.to(dtype=torch.float)

        return meta_task_graph1, meta_task_graph2

    def get_collate_fn(self):
        def aug_collate(batch):
            # Separating the subgraphs
            subgraph1_list, subgraph2_list = list(zip(*batch))

            # Creating two separate batches
            batch1 = Batch.from_data_list(subgraph1_list)
            batch2 = Batch.from_data_list(subgraph2_list)

            return batch1, batch2

        return aug_collate


class FewShotKGHierDataset(FewShotSubgraphDataset):
    def __init__(
        self,
        pyg_graph,
        class_emb,
        data_idx,
        n_way,
        k_shot,
        q_query,
        datamanager: FewShotDataManager,
        mode,
        edges,  # all edges in the graph
        fs_edges,  # edges that belongs to specific classes for few-shot
        fs_edge_types,
        hop=2,
        prompt_feat=None,
        to_undirected=False,
        single_prompt_edge=True,
        adj=None,
        **kwargs,
    ):
        self.g = pyg_graph
        self.kwargs = kwargs
        # mode 0 for sample index from training classes, 1 for val, 2 for test
        self.fs_idx_loader = datamanager.get_data_loader(mode)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.edges = edges
        self.fs_edges = fs_edges
        self.fs_edge_types = fs_edge_types
        self.prompt_feat = prompt_feat
        self.single_prompt_edge = single_prompt_edge

        edge_index = fs_edges
        if to_undirected:
            edge_index = pyg.utils.to_undirected(edge_index)

        if adj is not None:
            self.adj = adj
        else:
            self.adj = csr_array(
                (
                    torch.ones(len(edge_index[0])),
                    (edge_index[0], edge_index[1]),
                ),
                shape=(self.g.num_nodes, self.g.num_nodes),
            )
        self.class_emb = class_emb
        self.hop = hop
        self.data_idx = data_idx

    def get_single_subgraph(self, node_id):
        node_ids = self.edges[:, node_id]
        neighbors = sample_fixed_hop_size_neighbor(
            self.adj, node_ids, self.hop, max_nodes_per_hope=100
        )
        neighbors = np.r_[node_ids, neighbors]

        node_mask = torch.zeros(self.g.num_nodes, dtype=torch.bool)
        node_mask[neighbors] = True

        edge_mask = node_mask[self.fs_edges[0]] & node_mask[self.fs_edges[1]]

        edge2idx = torch.zeros(self.g.num_nodes, dtype=torch.long)
        edge2idx[neighbors] = torch.arange(len(neighbors))
        edge_index = self.fs_edges[:, edge_mask]
        edge_type = self.fs_edge_types[edge_mask]
        edge_index = edge2idx[edge_index]

        return neighbors, edge_index, neighbors.shape[0], edge_type

    def make_feature_graph(self, graph_set):
        """
        For each query node, combine its subgraph with all support subgraphs.
        Args:
            qry_subgraph:
            spt_subgraphs:

        Returns:

        """
        (qry_subgraph, spt_subgraphs) = graph_set
        neighbors, edge_index, num_nodes, edge_type = qry_subgraph
        neighbors_list = [neighbors] + [
            subgraph[0] for subgraph in spt_subgraphs
        ]
        feat = self.g.x_text_feat[np.concatenate(neighbors_list).astype(int)]
        if not isinstance(feat, torch.Tensor):
            feat = torch.from_numpy(feat)
            feat = feat.float()

        # nodes_pt represents the index of first node in each subgraph
        # nodes_pt[0] is the index of query node; nodes_pt[1,-1]: idx of support nodes; nodes_pt[-1]: idx of first prompt node
        nodes_pt = [0, num_nodes] + [subgraph[2] for subgraph in spt_subgraphs]
        nodes_pt = list(np.cumsum(nodes_pt))

        edge_index = torch.cat(
            [edge_index]
            + [
                spt_subgraphs[idx][1] + nodes_pt[idx + 1]
                for idx in range(len(spt_subgraphs))
            ],
            dim=1,
        )
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=-1)
        edge_type = torch.cat(
            [edge_type]
            + [spt_subgraphs[idx][3] for idx in range(len(spt_subgraphs))],
        )

        edge_feat = self.g.edge_text_feat[
            torch.cat(
                [edge_type, edge_type + int(len(self.g.edge_text_feat) / 2)]
            )
        ]

        return feat, edge_index, edge_feat, nodes_pt

    def make_prompted_graph(self, feature_graph):
        (
            node_cls,
            feat,
            edge_index,
            edge_feat,
            nodes_pt,
            label,
            true_class,
        ) = feature_graph

        assert nodes_pt[-1] == (feat.shape[0])
        true_edge_num = len(edge_index[0])
        cls_feat = self.class_emb[node_cls]
        feat = torch.cat(
            [
                feat,
                cls_feat,
                self.prompt_feat.view(1, -1).repeat(1 + self.n_way, 1),
            ]
        )

        new_node_id = 0

        # Connect support nodes with corresponding class node
        spt_pt = nodes_pt[1:-1]
        cls_pt = [
            nodes_pt[-1] + i + 1 + self.n_way
            for i in range(self.n_way)
            for j in range(self.k_shot)
        ]

        spt_edge = torch.tensor(
            [
                spt_pt + [pt + 1 for pt in spt_pt],
                cls_pt * 2,
            ],
            dtype=torch.long,
        )

        # Connect spt_h_nodes with h_nodes
        spt_h_nodes = [
            nodes_pt[-1] + 1 + self.n_way + i for i in range(self.n_way)
        ]
        prompt_nodes = [nodes_pt[-1] + i for i in range(self.n_way)]
        spt_prompt_edge = torch.tensor(
            [
                spt_h_nodes,
                prompt_nodes,
            ],
            dtype=torch.long,
        )

        # Connect qry_h_node
        qry_h_edge = torch.tensor(
            [
                [nodes_pt[-1] + self.n_way for i in range(self.n_way)]
                + [new_node_id, new_node_id + 1],
                [nodes_pt[-1] + i for i in range(self.n_way)]
                + [nodes_pt[-1] + self.n_way] * 2,
            ],
            dtype=torch.long,
        )

        # get final edge index and edge types
        if self.single_prompt_edge:
            edge_index = torch.cat(
                [
                    edge_index,
                    spt_edge,
                    spt_prompt_edge,
                    qry_h_edge,
                ],
                dim=-1,
            )
            e_type = torch.cat(
                [
                    torch.zeros(true_edge_num, dtype=torch.long),
                    torch.zeros(len(spt_edge[0]), dtype=torch.long) + 1,
                    torch.zeros(len(spt_prompt_edge[0]), dtype=torch.long) + 2,
                    torch.zeros(len(qry_h_edge[0]) - 2, dtype=torch.long) + 3,
                    torch.zeros(2, dtype=torch.long) + 4,
                ]
            )
            edge_feat = torch.cat(
                [
                    edge_feat,
                    self.g.prompt_edge_feat.repeat(
                        [
                            len(spt_prompt_edge[0])
                            + len(spt_edge[0])
                            + len(qry_h_edge[0]),
                            1,
                        ]
                    ),
                ]
            )
        assert edge_feat.size(0) == e_type.size(0)

        # get node masks
        new_subg = pyg.data.Data(
            feat, edge_index, y=label, edge_type=e_type, edge_attr=edge_feat
        )
        true_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        true_nodes_mask[nodes_pt[-1] : nodes_pt[-1] + self.n_way] = True
        assert (true_nodes_mask == True).sum() == self.n_way
        h_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        h_node_mask[nodes_pt[-1] + self.n_way :] = True
        target_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        target_node_mask[new_node_id] = True
        spt_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        spt_nodes_mask[nodes_pt[1:-1]] = True
        assert (spt_nodes_mask == True).sum() == self.n_way * self.k_shot

        new_subg.target_node_mask = target_node_mask
        new_subg.h_node_mask = h_node_mask
        new_subg.true_nodes_mask = true_nodes_mask
        new_subg.spt_nodes_mask = spt_nodes_mask
        new_subg.sample_num_nodes = new_subg.num_nodes
        new_subg.num_classes = self.n_way
        return new_subg


class FewShotKGDataset(FewShotKGHierDataset):
    def make_prompted_graph(self, feature_graph):
        (
            node_cls,
            feat,
            edge_index,
            edge_feat,
            nodes_pt,
            label,
            true_class,
        ) = feature_graph

        assert nodes_pt[-1] == (feat.shape[0])
        true_edge_num = len(edge_index[0])
        cls_feat = self.class_emb[node_cls]
        feat = torch.cat(
            [
                feat,
                cls_feat,
                self.prompt_feat.view(1, -1).repeat(
                    1 + self.n_way * self.k_shot, 1
                ),
            ]
        )

        new_node_id = 0

        # Connect support nodes with corresponding class node
        spt_pt = nodes_pt[1:-1]

        cls_pt = [
            nodes_pt[-1] + i + 1 + self.n_way
            for i in range(self.n_way)
            for j in range(self.k_shot)
        ]

        spt_edge = torch.tensor(
            [
                spt_pt + [pt + 1 for pt in spt_pt],
                cls_pt * 2,
            ],
            dtype=torch.long,
        )

        # Connect spt_h_nodes with h_nodes
        spt_h_nodes = [
            nodes_pt[-1] + 1 + self.n_way + i for i in range(self.n_way)
        ]
        prompt_nodes = [nodes_pt[-1] + i for i in range(self.n_way)]
        spt_prompt_edge = torch.tensor(
            [
                spt_h_nodes,
                prompt_nodes,
            ],
            dtype=torch.long,
        )

        # Connect qry_h_node
        qry_h_edge = torch.tensor(
            [
                [nodes_pt[-1] + self.n_way for i in range(self.n_way)]
                + [new_node_id, new_node_id + 1]
                + [nodes_pt[-1] + self.n_way] * 2,
                [nodes_pt[-1] + i for i in range(self.n_way)]
                + [nodes_pt[-1] + self.n_way] * 2
                + [new_node_id, new_node_id + 1],
            ],
            dtype=torch.long,
        )

        # get final edge index and edge types
        if self.single_prompt_edge:
            edge_index = torch.cat(
                [
                    edge_index,
                    spt_edge,
                    spt_edge[[1, 0]],
                    spt_prompt_edge,
                    qry_h_edge,
                ],
                dim=-1,
            )
            e_type = torch.cat(
                [
                    torch.zeros(true_edge_num, dtype=torch.long),
                    torch.zeros(len(spt_edge[0]), dtype=torch.long) + 1,
                    torch.zeros(len(spt_edge[0]), dtype=torch.long) + 2,
                    torch.zeros(len(spt_prompt_edge[0]), dtype=torch.long) + 4,
                    torch.zeros(len(qry_h_edge[0]) - 4, dtype=torch.long) + 3,
                    torch.zeros(2, dtype=torch.long) + 1,
                    torch.zeros(2, dtype=torch.long) + 2,
                ]
            )
            edge_feat = torch.cat(
                [
                    edge_feat,
                    self.g.prompt_edge_feat.repeat(
                        [
                            len(spt_prompt_edge[0])
                            + len(spt_edge[0]) * 2
                            + len(qry_h_edge[0]),
                            1,
                        ]
                    ),
                ]
            )
        assert edge_feat.size(0) == e_type.size(0)

        # get node masks
        new_subg = pyg.data.Data(
            feat, edge_index, y=label, edge_type=e_type, edge_attr=edge_feat
        )
        true_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        true_nodes_mask[nodes_pt[-1] : nodes_pt[-1] + self.n_way] = True
        assert (true_nodes_mask == True).sum() == self.n_way
        h_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        h_node_mask[nodes_pt[-1] + self.n_way :] = True
        target_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        target_node_mask[new_node_id] = True
        spt_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        spt_nodes_mask[nodes_pt[1:-1]] = True
        assert (spt_nodes_mask == True).sum() == self.n_way * self.k_shot

        new_subg.target_node_mask = target_node_mask
        new_subg.h_node_mask = h_node_mask
        new_subg.true_nodes_mask = true_nodes_mask
        new_subg.spt_nodes_mask = spt_nodes_mask
        new_subg.sample_num_nodes = new_subg.num_nodes
        new_subg.num_classes = self.n_way
        return new_subg


class ZeroShotKGDataset(FewShotKGHierDataset):
    def make_feature_graph(self, graph_set):
        (qry_subgraph, _) = graph_set
        neighbors, edge_index, num_nodes, edge_type = qry_subgraph
        feat = self.g.x_text_feat[neighbors.astype(int)]
        if not isinstance(feat, torch.Tensor):
            feat = torch.from_numpy(feat)
            feat = feat.float()

        nodes_pt = [0, num_nodes]
        edge_feat = self.g.edge_text_feat[
            torch.cat(
                [edge_type, edge_type + int(len(self.g.edge_text_feat) / 2)]
            )
        ]
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=-1)

        return feat, edge_index, edge_feat, nodes_pt

    def make_prompted_graph(self, feature_graph):
        (
            node_cls,
            feat,
            edge_index,
            edge_feat,
            nodes_pt,
            label,
            true_class,
        ) = feature_graph

        assert nodes_pt[-1] == (feat.shape[0])
        true_edge_num = len(edge_index[0])
        cls_feat = self.class_emb[node_cls]
        feat = torch.cat([feat, cls_feat, self.prompt_feat.view(1, -1)])

        new_node_id = 0

        # Connect qry_h_node
        qry_h_edge = torch.tensor(
            [
                [nodes_pt[-1] + self.n_way for i in range(self.n_way)]
                + [new_node_id, new_node_id + 1]
                + [nodes_pt[-1] + self.n_way] * 2,
                [nodes_pt[-1] + i for i in range(self.n_way)]
                + [nodes_pt[-1] + self.n_way] * 2
                + [new_node_id, new_node_id + 1],
            ],
            dtype=torch.long,
        )

        # get final edge index and edge types
        if self.single_prompt_edge:
            edge_index = torch.cat(
                [
                    edge_index,
                    qry_h_edge,
                ],
                dim=-1,
            )
            e_type = torch.cat(
                [
                    torch.zeros(true_edge_num, dtype=torch.long),
                    torch.zeros(len(qry_h_edge[0]) - 4, dtype=torch.long) + 3,
                    torch.zeros(2, dtype=torch.long) + 1,
                    torch.zeros(2, dtype=torch.long) + 2,
                ]
            )
            edge_feat = torch.cat(
                [
                    edge_feat,
                    self.g.prompt_edge_feat.repeat([len(qry_h_edge[0]), 1]),
                ]
            )
        assert edge_feat.size(0) == e_type.size(0)

        # get node masks
        new_subg = pyg.data.Data(
            feat, edge_index, y=label, edge_type=e_type, edge_attr=edge_feat
        )
        true_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        true_nodes_mask[nodes_pt[-1] : nodes_pt[-1] + self.n_way] = True
        assert (true_nodes_mask == True).sum() == self.n_way
        h_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        h_node_mask[nodes_pt[-1] + self.n_way :] = True
        target_node_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        target_node_mask[new_node_id] = True
        spt_nodes_mask = torch.zeros(new_subg.num_nodes, dtype=torch.bool)
        assert (spt_nodes_mask == True).sum() == self.n_way * self.k_shot

        new_subg.target_node_mask = target_node_mask
        new_subg.h_node_mask = h_node_mask
        new_subg.true_nodes_mask = true_nodes_mask
        new_subg.spt_nodes_mask = spt_nodes_mask
        new_subg.sample_num_nodes = new_subg.num_nodes
        new_subg.num_classes = self.n_way
        return new_subg


class GraphListDataset(GraphTextDataset):
    def __init__(
        self,
        graphs,
        class_embs,
        prompt_edge_feat,
        data_idx,
        trim_class_func=None,
        single_prompt_edge=False,
        **kwargs,
    ):
        super().__init__(graphs, **kwargs)
        self.class_embs = class_embs
        self.prompt_edge_feat = prompt_edge_feat
        self.data_idx = data_idx
        self.trim_class_func = trim_class_func
        self.single_prompt_edge = single_prompt_edge

    def __len__(self):
        return len(self.data_idx)

    def trim_class(self, label):
        if self.trim_class_func is None:
            return label, self.class_embs, label
        else:
            return self.trim_class_func(self.class_embs, label)

    def make_feature_graph(self, index):
        g = self.g[self.data_idx[index]]
        edge_index = g.edge_index
        label = g.y
        # label_emb = self.class_emb(label).view(1, -1)
        feat = g.x_text_feat
        next_nid = g.num_nodes
        edge_feat = g.xe_text_feat
        y_label, g_class_emb, trimmed_label = self.trim_class(label)
        return (
            feat,
            edge_feat,
            edge_index,
            next_nid,
            g_class_emb,
            y_label,
            trimmed_label,
        )

    def make_prompted_graph(self, feature_graph):
        (
            feat,
            edge_feat,
            edge_index,
            next_nid,
            g_class_emb,
            label,
            trimmed_label,
        ) = feature_graph
        feat = torch.cat([feat, g_class_emb], dim=0)
        virtual_edge = torch.stack(
            [
                torch.arange(next_nid, dtype=torch.long)
                .repeat(1, len(g_class_emb))
                .view(-1),
                torch.arange(
                    next_nid, next_nid + len(g_class_emb), dtype=torch.long
                ).repeat_interleave(next_nid),
            ],
            dim=0,
        )
        # print(virtual_edge)
        edge_index = torch.cat(
            [
                edge_index,
                virtual_edge,
                # virtual_edge[[1, 0]],
            ],
            dim=-1,
        )
        edge_type = torch.cat(
            [
                torch.zeros(len(edge_feat), dtype=torch.long),
                torch.zeros(len(virtual_edge[0]), dtype=torch.long) + 1,
                # torch.zeros(len(virtual_edge[0]), dtype=torch.long) + 2,
            ]
        )
        edge_emb = torch.cat(
            [
                edge_feat,
                self.prompt_edge_feat[0].repeat(
                    [(next_nid) * len(g_class_emb), 1]
                ),
            ]
        )
        prompted_graph = pyg.data.Data(
            feat,
            edge_index,
            y=label,
            edge_attr=edge_emb,
            edge_type=edge_type,
        )
        # print(prompted_graph)
        # print(edge_index)
        return prompted_graph

    def to_pyg(self, feature_graph, prompted_graph):
        true_nodes_mask = torch.zeros(
            prompted_graph.num_nodes, dtype=torch.bool
        )
        true_nodes_mask[
            prompted_graph.num_nodes - len(feature_graph[-3]) :
        ] = True
        bin_labels = torch.zeros(prompted_graph.num_nodes, dtype=torch.float)
        bin_labels[
            prompted_graph.num_nodes - len(feature_graph[-3]) :
        ] = feature_graph[-1]
        # noi_node_mask = torch.zeros(prompted_graph.num_nodes, dtype=torch.bool)
        # noi_node_mask[
        #     prompted_graph.num_nodes - len(feature_graph[-3]) - 1
        # ] = True
        # prompted_graph.noi_node_mask = noi_node_mask
        # if self.trim_class_func is None:
        prompted_graph.bin_labels = bin_labels
        target_node_mask = torch.zeros(
            prompted_graph.num_nodes, dtype=torch.bool
        )
        target_node_mask[: feature_graph[-4]] = True
        prompted_graph.target_node_mask = target_node_mask
        prompted_graph.true_nodes_mask = true_nodes_mask
        prompted_graph.sample_num_nodes = prompted_graph.num_nodes
        prompted_graph.num_classes = len(feature_graph[-3])
        return prompted_graph


class GraphListNopromptDataset(GraphListDataset):
    def make_prompted_graph(self, feature_graph):
        (
            feat,
            edge_feat,
            edge_index,
            next_nid,
            g_class_emb,
            label,
            trimmed_label,
        ) = feature_graph
        feat = torch.cat([feat, g_class_emb], dim=0)
        edge_type = torch.zeros(len(edge_feat), dtype=torch.long)
        prompted_graph = pyg.data.Data(
            feat,
            edge_index,
            y=label,
            edge_attr=edge_feat,
            edge_type=edge_type,
        )
        # print(prompted_graph)
        # print(edge_index)
        return prompted_graph


class GraphListHierDataset(GraphListDataset):
    def __init__(
        self,
        graphs,
        class_embs,
        prompt_edge_feat,
        prompt_text_feat,
        data_idx,
        trim_class_func=None,
        single_prompt_edge=False,
        **kwargs,
    ):
        super().__init__(
            graphs,
            class_embs,
            prompt_edge_feat,
            data_idx,
            trim_class_func,
            single_prompt_edge,
            **kwargs,
        )
        self.prompt_text_feat = prompt_text_feat

    def make_prompted_graph(self, feature_graph):
        (
            feat,
            edge_feat,
            edge_index,
            next_nid,
            g_class_emb,
            label,
            trimmed_label,
        ) = feature_graph
        feat = torch.cat([feat, self.prompt_text_feat[:1], g_class_emb], dim=0)
        feature2prompt_edge = torch.tensor(
            [list(range(next_nid)), [next_nid] * next_nid],
            dtype=torch.long,
        )
        prompt2prompt_edge = torch.tensor(
            [
                [next_nid] * len(g_class_emb),
                [next_nid + i + 1 for i in range(len(g_class_emb))],
            ],
            dtype=torch.long,
        )
        prompt_edges = [
            feature2prompt_edge,
            feature2prompt_edge[[1, 0]],
            prompt2prompt_edge,
        ]
        prompt_edge_types = [
            torch.zeros(
                len(feature2prompt_edge[0]),
                dtype=torch.long,
            )
            + 1,
            torch.zeros(len(feature2prompt_edge[0]), dtype=torch.long) + 2,
            torch.zeros(len(prompt2prompt_edge[0]), dtype=torch.long) + 3,
        ]
        prompt_feat_multiple = 1
        if not self.single_prompt_edge:
            prompt_edges += [prompt2prompt_edge[[1, 0]]]
            prompt_edge_types += [
                torch.zeros(len(prompt2prompt_edge[0]), dtype=torch.long) + 4
            ]
            prompt_feat_multiple = 2
        edge_index = torch.cat([edge_index] + prompt_edges, dim=-1)
        edge_type = torch.cat(
            [torch.zeros(len(edge_feat), dtype=torch.long)] + prompt_edge_types
        )
        edge_feat = torch.cat(
            [
                edge_feat,
                self.prompt_edge_feat[0].repeat(
                    [
                        2 * len(feature2prompt_edge[0])
                        + len(prompt2prompt_edge[0]) * prompt_feat_multiple,
                        1,
                    ]
                ),
            ]
        )
        prompted_graph = pyg.data.Data(
            feat,
            edge_index,
            y=label,
            edge_attr=edge_feat,
            edge_type=edge_type,
        )
        return prompted_graph


class GraphListHierFSDataset(GraphListDataset):
    def __init__(
        self,
        graphs,
        class_embs,
        prompt_edge_feat,
        prompt_text_feat,
        data_idx,
        trim_class_func=None,
        single_prompt_edge=False,
        class_ind=None,
        max_sample=10,
        shot=None,
        reuse=False,
        use_class_emb=False,
        target_class=None,
        **kwargs,
    ):
        super().__init__(
            graphs,
            class_embs,
            prompt_edge_feat,
            data_idx,
            trim_class_func,
            single_prompt_edge,
            **kwargs,
        )
        self.prompt_text_feat = prompt_text_feat
        self.class_ind = class_ind
        self.max_sample = max_sample
        self.num_classes = self.class_ind.size()[1]
        self.shot = shot
        self.reuse = reuse
        self.true_can = [
            (can == 1).nonzero(as_tuple=True)[0] for can in self.class_ind.T
        ]
        self.false_can = [
            (can == 0).nonzero(as_tuple=True)[0] for can in self.class_ind.T
        ]
        self.use_class_emb = use_class_emb
        self.target_class = target_class

    def make_feature_graph(self, index):
        if self.target_class is not None:
            class_ind = self.target_class[
                torch.randint(0, len(self.target_class), (1,))
            ]
        else:
            class_ind = torch.randint(0, self.num_classes, (1,))
        if self.shot is None:
            while (
                len(self.true_can[class_ind]) <= 1
                or len(self.false_can[class_ind]) <= 1
            ):
                class_ind = torch.randint(0, self.num_classes, (1,))
            c_max_sample = min(
                len(self.true_can[class_ind]), len((self.false_can[class_ind]))
            )
            num_sample = torch.randint(
                2, min(c_max_sample, self.max_sample) + 1, (1,)
            )
        else:
            while (
                len(self.true_can[class_ind]) <= self.shot
                or len(self.false_can[class_ind]) <= self.shot
            ):
                class_ind = torch.randint(0, self.num_classes, (1,))
            num_sample = self.shot + 1
        # print(num_sample)
        true_can = self.true_can[class_ind][
            torch.randperm(len(self.true_can[class_ind]))[:num_sample]
        ]
        # print(true_can)
        false_can = self.false_can[class_ind][
            torch.randperm(len(self.false_can[class_ind]))[:num_sample]
        ]
        true_feature_graph = [
            super(GraphListHierFSDataset, self).make_feature_graph(can)
            for can in true_can
        ]
        false_feature_graph = [
            super(GraphListHierFSDataset, self).make_feature_graph(can)
            for can in false_can
        ]
        return (
            true_feature_graph,
            false_feature_graph,
            self.class_embs[class_ind],
        )

    def self_sample_edge_index(self, init_ind, size):
        prompt_init_ind = init_ind + size * 2
        self_edges = torch.stack(
            [
                torch.arange(init_ind, prompt_init_ind),
                torch.arange(prompt_init_ind, prompt_init_ind + size * 2),
            ],
            dim=0,
        )
        fs_edges = []
        for i in range(size):
            fs_edges.append(
                torch.stack(
                    [
                        i
                        + init_ind
                        + torch.zeros(2 * size - 2, dtype=torch.long),
                        torch.cat(
                            [
                                torch.arange(
                                    prompt_init_ind,
                                    i + prompt_init_ind,
                                    dtype=torch.long,
                                ),
                                torch.arange(
                                    i + prompt_init_ind + 1,
                                    prompt_init_ind + size,
                                    dtype=torch.long,
                                ),
                                torch.arange(
                                    prompt_init_ind + size,
                                    prompt_init_ind + size + i,
                                    dtype=torch.long,
                                ),
                                torch.arange(
                                    prompt_init_ind + size + i + 1,
                                    prompt_init_ind + 2 * size,
                                    dtype=torch.long,
                                ),
                            ]
                        ),
                    ],
                    dim=0,
                )
            )
        fs_edges = torch.cat(fs_edges, dim=-1)
        return self_edges, fs_edges

    def make_prompted_graph(self, feature_graph):
        if self.reuse:
            return self.make_prompted_graph_reuse(feature_graph)
        else:
            return self.make_prompted_graph_simple(feature_graph)

    def make_prompted_graph_simple(self, feature_graph):
        (true_feature_graph, false_feature_graph, class_emb) = feature_graph
        if torch.rand(1)[0] < 0.5:
            target_graph = true_feature_graph[0]
            sample_label = torch.tensor([[1]])
        else:
            target_graph = false_feature_graph[0]
            sample_label = torch.tensor([[0]])

        all_graphs = [target_graph] + true_feature_graph[1:]
        num_nodes = torch.tensor(
            [len(g[0]) for g in all_graphs], dtype=torch.long
        )
        total_num_nodes = num_nodes.sum()
        feat = torch.cat(
            [g[0] for g in all_graphs]
            + [
                self.prompt_text_feat[:1].repeat(len(all_graphs), 1),
                # class_emb,
                class_emb
                if self.use_class_emb
                else self.prompt_text_feat[1:2].repeat(len(class_emb), 1),
            ],
            dim=0,
        )

        feature_nodes_indices = torch.arange(total_num_nodes, dtype=torch.long)
        prompt_nodes_indices = torch.arange(
            total_num_nodes,
            total_num_nodes + len(all_graphs),
            dtype=torch.long,
        ).repeat_interleave(num_nodes)

        feature2prompt_edge = torch.stack(
            [feature_nodes_indices, prompt_nodes_indices], dim=0
        )
        valid_label_ind = total_num_nodes + len(all_graphs)
        query_prompt_edge = torch.stack(
            [
                torch.zeros(1, dtype=torch.long) + total_num_nodes,
                torch.arange(
                    total_num_nodes + len(all_graphs),
                    total_num_nodes + len(all_graphs) + 1,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        support_prompt_edge = torch.stack(
            [
                torch.arange(
                    total_num_nodes + 1,
                    total_num_nodes + len(all_graphs),
                    dtype=torch.long,
                ),
                valid_label_ind.repeat_interleave(len(all_graphs) - 1),
            ],
            dim=0,
        )

        prompt2prompt_edge = torch.cat(
            [query_prompt_edge, support_prompt_edge],
            dim=-1,
        )
        prompt_edges = [
            feature2prompt_edge,
            feature2prompt_edge[[1, 0]],
            prompt2prompt_edge,
        ]
        prompt_edge_types = [
            torch.zeros(
                len(feature2prompt_edge[0]),
                dtype=torch.long,
            )
            + 1,
            torch.zeros(len(feature2prompt_edge[0]), dtype=torch.long) + 2,
            torch.zeros(len(query_prompt_edge[0]), dtype=torch.long) + 3,
            torch.zeros(len(support_prompt_edge[0]), dtype=torch.long) + 4,
        ]
        prompt_feat_multiple = 1
        if not self.single_prompt_edge:
            prompt_edges += [prompt2prompt_edge[[1, 0]]]
            prompt_edge_types += [
                torch.zeros(len(prompt2prompt_edge[0]), dtype=torch.long) + 4
            ]
            prompt_feat_multiple = 2
        edge_feat = torch.cat([g[1] for g in all_graphs], dim=0)
        edge_index = torch.cat([g[2] for g in all_graphs], dim=-1)
        edge_index = torch.cat([edge_index] + prompt_edges, dim=-1)
        edge_type = torch.cat(
            [torch.zeros(len(edge_feat), dtype=torch.long)] + prompt_edge_types
        )
        edge_feat = torch.cat(
            [
                edge_feat,
                self.prompt_edge_feat[0].repeat(
                    [
                        2 * len(feature2prompt_edge[0]),
                        1,
                    ]
                ),
                self.prompt_edge_feat[1].repeat(
                    [
                        len(query_prompt_edge[0]),
                        1,
                    ]
                ),
                self.prompt_edge_feat[2].repeat(
                    [
                        len(support_prompt_edge[0]),
                        1,
                    ]
                ),
            ]
        )
        prompted_graph = pyg.data.Data(
            feat,
            edge_index,
            y=sample_label,
            edge_attr=edge_feat,
            edge_type=edge_type,
        )
        return prompted_graph

    def make_prompted_graph_reuse(self, feature_graph):
        (true_feature_graph, false_feature_graph, class_emb) = feature_graph
        num_sample = len(false_feature_graph)
        all_graphs = true_feature_graph + false_feature_graph
        num_nodes = torch.tensor(
            [len(g[0]) for g in all_graphs], dtype=torch.long
        )
        total_num_nodes = num_nodes.sum()
        # print(self.prompt_text_feat)
        feat = torch.cat(
            [g[0] for g in all_graphs]
            + [
                self.prompt_text_feat.repeat(len(all_graphs), 1),
                class_emb.repeat(len(all_graphs), 1),
            ],
            dim=0,
        )
        feature_nodes_indices = torch.arange(total_num_nodes, dtype=torch.long)
        prompt_nodes_indices = torch.arange(
            total_num_nodes,
            total_num_nodes + len(all_graphs),
            dtype=torch.long,
        ).repeat_interleave(num_nodes)
        feature2prompt_edge = torch.stack(
            [feature_nodes_indices, prompt_nodes_indices], dim=0
        )
        (
            self_prompt2prompt_edge,
            fs_prompt2prompt_edge,
        ) = self.self_sample_edge_index(total_num_nodes, num_sample)
        prompt2prompt_edge = torch.cat(
            [self_prompt2prompt_edge, fs_prompt2prompt_edge],
            dim=-1,
        )
        prompt_edges = [
            feature2prompt_edge,
            feature2prompt_edge[[1, 0]],
            prompt2prompt_edge,
        ]
        prompt_edge_types = [
            torch.zeros(
                len(feature2prompt_edge[0]),
                dtype=torch.long,
            )
            + 1,
            torch.zeros(len(feature2prompt_edge[0]), dtype=torch.long) + 2,
            torch.zeros(len(prompt2prompt_edge[0]), dtype=torch.long) + 3,
        ]
        prompt_feat_multiple = 1
        if not self.single_prompt_edge:
            prompt_edges += [prompt2prompt_edge[[1, 0]]]
            prompt_edge_types += [
                torch.zeros(len(prompt2prompt_edge[0]), dtype=torch.long) + 4
            ]
            prompt_feat_multiple = 2
        edge_feat = torch.cat([g[1] for g in all_graphs], dim=0)
        edge_index = torch.cat([g[2] for g in all_graphs], dim=-1)
        edge_index = torch.cat([edge_index] + prompt_edges, dim=-1)
        edge_type = torch.cat(
            [torch.zeros(len(edge_feat), dtype=torch.long)] + prompt_edge_types
        )
        edge_feat = torch.cat(
            [
                edge_feat,
                self.prompt_edge_feat[0].repeat(
                    [
                        2 * len(feature2prompt_edge[0]),
                        1,
                    ]
                ),
                self.prompt_edge_feat[1].repeat(
                    [
                        len(self_prompt2prompt_edge[0]),
                        1,
                    ]
                ),
                self.prompt_edge_feat[2].repeat(
                    [
                        len(fs_prompt2prompt_edge[0]),
                        1,
                    ]
                ),
            ]
        )
        prompted_graph = pyg.data.Data(
            feat,
            edge_index,
            y=torch.cat([g[-2] for g in all_graphs], dim=0),
            edge_attr=edge_feat,
            edge_type=edge_type,
        )
        return prompted_graph

    def to_pyg(self, feature_graph, prompted_graph):
        if self.reuse:
            return self.to_pyg_reuse(feature_graph, prompted_graph)
        else:
            return self.to_pyg_simple(feature_graph, prompted_graph)

    def to_pyg_reuse(self, feature_graph, prompted_graph):
        true_nodes_mask = torch.zeros(
            prompted_graph.num_nodes, dtype=torch.bool
        )
        true_nodes_mask[
            prompted_graph.num_nodes - len(feature_graph[1]) * 2 :
        ] = True
        bin_labels = torch.zeros(prompted_graph.num_nodes, dtype=torch.float)
        bin_labels[
            prompted_graph.num_nodes
            - len(feature_graph[1]) * 2 : prompted_graph.num_nodes
            - len(feature_graph[1])
        ] = 1
        # bin_labels[prompted_graph.num_nodes - len(feature_graph[1]) :] = 0
        # if self.trim_class_func is None:
        prompted_graph.bin_labels = bin_labels
        target_node_mask = torch.zeros(
            prompted_graph.num_nodes, dtype=torch.bool
        )
        # target_node_mask[feature_graph[0][-4]] = True
        prompted_graph.target_node_mask = target_node_mask
        prompted_graph.true_nodes_mask = true_nodes_mask
        prompted_graph.sample_num_nodes = prompted_graph.num_nodes
        prompted_graph.num_classes = 1
        return prompted_graph

    def to_pyg_simple(self, feature_graph, prompted_graph):
        true_nodes_mask = torch.zeros(
            prompted_graph.num_nodes, dtype=torch.bool
        )
        true_nodes_mask[prompted_graph.num_nodes - 1 :] = True
        bin_labels = torch.zeros(prompted_graph.num_nodes, dtype=torch.float)
        bin_labels[prompted_graph.num_nodes - 1 :] = prompted_graph.y
        # bin_labels[prompted_graph.num_nodes - len(feature_graph[1]) :] = 0
        # if self.trim_class_func is None:
        prompted_graph.bin_labels = bin_labels
        target_node_mask = torch.zeros(
            prompted_graph.num_nodes, dtype=torch.bool
        )
        # target_node_mask[feature_graph[0][-4]] = True
        prompted_graph.target_node_mask = target_node_mask
        prompted_graph.true_nodes_mask = true_nodes_mask
        prompted_graph.sample_num_nodes = prompted_graph.num_nodes
        prompted_graph.num_classes = 1
        return prompted_graph


class GraphListHierFixDataset(GraphListDataset):
    def __init__(
        self,
        graphs,
        class_embs,
        prompt_edge_feat,
        prompt_text_feat,
        data_idx,
        trim_class_func=None,
        single_prompt_edge=False,
        class_ind=None,
        shot=10,
        use_class_emb=False,
        **kwargs,
    ):
        super().__init__(
            graphs,
            class_embs,
            prompt_edge_feat,
            data_idx,
            trim_class_func,
            single_prompt_edge,
            **kwargs,
        )
        self.prompt_text_feat = prompt_text_feat
        self.class_ind = class_ind
        self.shot = shot
        self.num_classes = self.class_ind.size()[1]
        self.true_can = [
            (can == 1).nonzero(as_tuple=True)[0] for can in self.class_ind.T
        ]
        self.false_can = [
            (can == 0).nonzero(as_tuple=True)[0] for can in self.class_ind.T
        ]
        self.use_class_emb = use_class_emb

    def make_feature_graph(self, index):
        query_graph = super(GraphListHierFixDataset, self).make_feature_graph(
            index
        )
        # print(true_can)
        label = query_graph[-1]
        support_graph = []
        valid_labels = []
        for i, label_val in enumerate(label.view(-1)):
            # print(i)
            if label_val == label_val:
                true_can = self.true_can[i]
                if len(true_can) < self.shot:
                    support_graph.append([])
                    label_val = float("nan")
                else:
                    support_g = true_can[
                        torch.randperm(len(true_can))[: self.shot]
                    ]
                    support_graph.append(
                        [
                            super(
                                GraphListHierFixDataset, self
                            ).make_feature_graph(g_ind)
                            for g_ind in support_g
                        ]
                    )
            valid_labels.append(label_val)
        true_label = torch.tensor(valid_labels, dtype=torch.float).view(1, -1)
        return (
            query_graph,
            support_graph,
            query_graph[-3],
            true_label,
        )

    def make_prompted_graph(self, feature_graph):
        (query_graph, support_graph, class_emb, true_label) = feature_graph
        all_graphs = [query_graph]
        for sg in support_graph:
            all_graphs += sg
        num_nodes = torch.tensor(
            [len(g[0]) for g in all_graphs], dtype=torch.long
        )
        total_num_nodes = num_nodes.sum()
        feat = torch.cat(
            [g[0] for g in all_graphs]
            + [
                self.prompt_text_feat[:1].repeat(len(all_graphs), 1),
                class_emb
                if self.use_class_emb
                else self.prompt_text_feat[1:2].repeat(len(class_emb), 1),
            ],
            dim=0,
        )
        feature_nodes_indices = torch.arange(total_num_nodes, dtype=torch.long)
        prompt_nodes_indices = torch.arange(
            total_num_nodes,
            total_num_nodes + len(all_graphs),
            dtype=torch.long,
        ).repeat_interleave(num_nodes)

        feature2prompt_edge = torch.stack(
            [feature_nodes_indices, prompt_nodes_indices], dim=0
        )
        valid_label_ind = (
            (true_label == true_label).nonzero(as_tuple=True)[1]
            + total_num_nodes
            + len(all_graphs)
        )
        query_prompt_edge = torch.stack(
            [
                torch.zeros(len(class_emb), dtype=torch.long)
                + total_num_nodes,
                torch.arange(
                    total_num_nodes + len(all_graphs),
                    total_num_nodes + len(all_graphs) + len(class_emb),
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        support_prompt_edge = torch.stack(
            [
                torch.arange(
                    total_num_nodes + 1,
                    total_num_nodes + len(all_graphs),
                    dtype=torch.long,
                ),
                valid_label_ind.repeat_interleave(self.shot),
            ],
            dim=0,
        )

        prompt2prompt_edge = torch.cat(
            [query_prompt_edge, support_prompt_edge],
            dim=-1,
        )
        prompt_edges = [
            feature2prompt_edge,
            feature2prompt_edge[[1, 0]],
            prompt2prompt_edge,
        ]
        prompt_edge_types = [
            torch.zeros(
                len(feature2prompt_edge[0]),
                dtype=torch.long,
            )
            + 1,
            torch.zeros(len(feature2prompt_edge[0]), dtype=torch.long) + 2,
            torch.zeros(len(query_prompt_edge[0]), dtype=torch.long) + 3,
            torch.zeros(len(support_prompt_edge[0]), dtype=torch.long) + 3,
        ]
        prompt_feat_multiple = 1
        if not self.single_prompt_edge:
            prompt_edges += [prompt2prompt_edge[[1, 0]]]
            prompt_edge_types += [
                torch.zeros(len(prompt2prompt_edge[0]), dtype=torch.long) + 4
            ]
            prompt_feat_multiple = 2
        edge_feat = torch.cat([g[1] for g in all_graphs], dim=0)
        edge_index = torch.cat([g[2] for g in all_graphs], dim=-1)
        edge_index = torch.cat([edge_index] + prompt_edges, dim=-1)
        edge_type = torch.cat(
            [torch.zeros(len(edge_feat), dtype=torch.long)] + prompt_edge_types
        )
        edge_feat = torch.cat(
            [
                edge_feat,
                self.prompt_edge_feat[0].repeat(
                    [
                        2 * len(feature2prompt_edge[0]),
                        1,
                    ]
                ),
                self.prompt_edge_feat[1].repeat(
                    [
                        len(query_prompt_edge[0]),
                        1,
                    ]
                ),
                self.prompt_edge_feat[2].repeat(
                    [
                        len(support_prompt_edge[0]),
                        1,
                    ]
                ),
            ]
        )
        prompted_graph = pyg.data.Data(
            feat,
            edge_index,
            y=torch.cat([g[-2] for g in all_graphs], dim=0),
            edge_attr=edge_feat,
            edge_type=edge_type,
        )
        return prompted_graph

    def to_pyg(self, feature_graph, prompted_graph):
        true_nodes_mask = torch.zeros(
            prompted_graph.num_nodes, dtype=torch.bool
        )
        true_nodes_mask[
            prompted_graph.num_nodes - len(feature_graph[-2]) :
        ] = True
        bin_labels = torch.zeros(prompted_graph.num_nodes, dtype=torch.float)
        bin_labels[
            prompted_graph.num_nodes - len(feature_graph[-2]) :
        ] = feature_graph[-1]
        # bin_labels[prompted_graph.num_nodes - len(feature_graph[1]) :] = 0
        # if self.trim_class_func is None:
        prompted_graph.bin_labels = bin_labels
        target_node_mask = torch.zeros(
            prompted_graph.num_nodes, dtype=torch.bool
        )
        # target_node_mask[feature_graph[0][-4]] = True
        prompted_graph.target_node_mask = target_node_mask
        prompted_graph.true_nodes_mask = true_nodes_mask
        prompted_graph.sample_num_nodes = prompted_graph.num_nodes
        prompted_graph.num_classes = len(feature_graph[-2])
        return prompted_graph


class MultiDataset(DatasetWithCollate):
    def __init__(
        self,
        datas,
        dataset_multiple=1,
        window_size=3,
        patience=3,
        min_ratio=0.1,
        # walk_length=None,
    ):
        self.datas = datas
        self.sizes = np.array([len(d) for d in datas])
        self.performance_record = []
        self.patience = patience
        if isinstance(self.patience, int):
            self.patience = np.zeros(len(self.sizes)) + self.patience
        self.inpatience = np.zeros(len(self.patience))
        self.window_size = window_size
        if isinstance(self.window_size, int):
            self.window_size = np.zeros(len(self.sizes)) + self.window_size
        self.dataset_multiple = dataset_multiple
        if not isinstance(self.dataset_multiple, list):
            self.dataset_multiple = (
                np.zeros(len(self.sizes)) + self.dataset_multiple
            )
        self.min_ratio = min_ratio
        if isinstance(self.min_ratio, float):
            self.min_ratio = np.zeros(len(self.sizes)) + self.min_ratio
        # self.walk_length = walk_length
        self.compute_sizes()

    def compute_sizes(self):
        self.aug_sizes = (self.sizes * np.array(self.dataset_multiple)).astype(
            int
        )
        self.size_seg = np.cumsum(self.aug_sizes)
        self.ind2dataset = np.arange(len(self.datas)).repeat(self.aug_sizes)
        self.sample_ind = (
            np.random.rand(len(self.ind2dataset))
            * self.sizes.repeat(self.aug_sizes)
        ).astype(int)
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
        for i in range(len(self.datas)):
            if len(self.performance_record) < self.window_size[i]:
                continue
            vals = [entry[i] for entry in self.performance_record][
                -int(self.window_size[i]) :
            ]
            vals = np.array(vals)
            mean = vals.mean()
            if metric[i] > mean:
                self.inpatience[i] = 0
            else:
                self.inpatience[i] += 1
            if self.inpatience[i] > self.patience[i]:
                self.dataset_multiple[i] = max(
                    self.min_ratio[i], self.dataset_multiple[i] / 2
                )
                # self.inpatience[i] = 0
        self.compute_sizes()
        self.performance_record.append(metric)
