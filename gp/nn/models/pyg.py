from gp.nn.models.GNN import MultiLayerMessagePassing

from gp.nn.models.util_model import MLP
from torch_geometric.nn.conv import GINConv, GINEConv, RGCNConv
import torch


class PyGGIN(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers,
        inp_dim,
        out_dim,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.build_layers()

    def build_input_layer(self):
        return GINConv(
            MLP(
                [self.inp_dim, 2 * self.inp_dim, self.out_dim],
                batch_norm=self.batch_norm is not None,
            ),
            train_eps=True,
        )

    def build_hidden_layer(self):
        return GINConv(
            MLP(
                [self.out_dim, 2 * self.out_dim, self.out_dim],
                batch_norm=self.batch_norm is not None,
            ),
            train_eps=True,
        )

    def build_message_from_input(self, g):
        return {"g": g.edge_index, "h": g.x}

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h}

    def layer_forward(self, layer, message):
        return self.conv[layer](message["h"], message["g"])


class PyGGINE(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers,
        inp_dim,
        out_dim,
        edge_dim,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.edge_dim = edge_dim
        self.build_layers()

    def build_input_layer(self):
        return GINEConv(
            MLP([self.inp_dim, self.inp_dim * 2, self.out_dim]),
            train_eps=True,
            edge_dim=self.edge_dim,
        )

    def build_hidden_layer(self):
        return GINEConv(
            MLP([self.out_dim, self.out_dim * 2, self.out_dim]),
            train_eps=True,
            edge_dim=self.edge_dim,
        )

    def build_message_from_input(self, g):
        return {"g": g.edge_index, "h": g.x, "e": g.edge_attr}

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": g.x, "e": g.edge_attr}

    def layer_forward(self, layer, message):
        return self.conv[layer](message["h"], message["g"], message["e"])


class PyGRGCN(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers,
        num_rels,
        inp_dim,
        out_dim,
        num_bases=None,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels
        self.num_bases = num_bases
        # self.layer_norms = torch.nn.ModuleList(
        #     [torch.nn.LayerNorm(out_dim) for _ in range(num_layers)])  # <-- Define LayerNorm modules
        self.build_layers()

    def build_input_layer(self):
        return RGCNConv(
            self.inp_dim, self.out_dim, self.num_rels, num_bases=self.num_bases
        )

    def build_hidden_layer(self):
        return RGCNConv(
            self.out_dim, self.out_dim, self.num_rels, num_bases=self.num_bases
        )

    def build_message_from_input(self, g):
        return {"g": g.edge_index, "h": g.x, "e": g.edge_type}

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h, "e": g.edge_type}

    def layer_forward(self, layer, message):
        # h = self.conv[layer](message["h"], message["g"], message["e"])
        # h = self.layer_norms[layer](h)
        # return h
        return self.conv[layer](message["h"], message["g"], message["e"])
