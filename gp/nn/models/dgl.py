import torch
from gp.nn.models.GNN import MultiLayerMessagePassing

from gp.nn.models.util_model import MLP
from dgl.nn.pytorch.conv import GINConv, RelGraphConv


class DGLGIN(MultiLayerMessagePassing):
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
            learn_eps=True,
        )

    def build_hidden_layer(self):
        return GINConv(
            MLP(
                [self.out_dim, 2 * self.out_dim, self.out_dim],
                batch_norm=self.batch_norm is not None,
            ),
            learn_eps=True,
        )

    def build_message_from_input(self, g, input_feat="feat"):
        if isinstance(input_feat, str):
            h = g.ndata[input_feat]
        elif torch.is_tensor(input_feat):
            h = input_feat
        else:
            raise NotImplementedError("Not supported input type")
        return {"g": g, "h": h}

    def build_message_from_output(self, g, h):
        return {"g": g, "h": h}

    def layer_forward(self, layer, message):
        return self.conv[layer](message["g"], message["h"])


class DGLRGCN(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers,
        num_rels,
        inp_dim,
        out_dim,
        num_bases=4,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.build_layers()

    def build_input_layer(self):
        return RelGraphConv(
            self.inp_dim, self.out_dim, self.num_rels, num_bases=self.num_bases
        )

    def build_hidden_layer(self):
        return RelGraphConv(
            self.out_dim, self.out_dim, self.num_rels, num_bases=self.num_bases
        )

    def build_message_from_input(self, g):
        return {"g": g, "h": g.ndata["feat"], "e": g.edata["type"]}

    def build_message_from_output(self, g, h):
        return {"g": g, "h": h, "e": g.edata["type"]}

    def layer_forward(self, layer, message):
        return self.conv[layer](message["g"], message["h"], message["e"])
