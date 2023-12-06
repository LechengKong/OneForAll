import torch
from torch import Tensor
from torch import nn

from gp.nn.models.util_model import MLP
from gp.nn.models.GNN import MultiLayerMessagePassing
from gp.nn.layer.pyg import RGCNEdgeConv, RGATEdgeConv
from torch_geometric.nn.pool import global_add_pool

from torch_geometric.transforms.add_positional_encoding import AddRandomWalkPE


class TextClassModel(torch.nn.Module):
    def __init__(self, model, outdim, task_dim, emb=None):
        super().__init__()
        self.model = model
        if emb is not None:
            self.emb = torch.nn.Parameter(emb.clone())

        self.mlp = MLP([2 * outdim, 2 * outdim, outdim, task_dim])

    def forward(self, g):
        emb = self.model(g)
        class_emb = emb[g.target_node_mask]
        att_emb = class_emb.repeat_interleave(len(self.emb), dim=0)
        att_emb = torch.cat(
            [att_emb, self.emb.repeat(len(class_emb), 1)], dim=-1
        )
        res = self.mlp(att_emb).view(-1, len(self.emb))
        return res


class AdaPoolClassModel(torch.nn.Module):
    def __init__(self, model, outdim, task_dim, emb=None):
        super().__init__()
        self.model = model
        if emb is not None:
            self.emb = torch.nn.Parameter(emb.clone())

        self.mlp = MLP([2 * outdim, 2 * outdim, outdim, task_dim])

    def forward(self, g):
        emb = self.model(g)
        float_mask = g.target_node_mask.to(torch.float)
        target_emb = float_mask.view(-1, 1) * emb
        n_count = global_add_pool(float_mask, g.batch, g.num_graphs)
        class_emb = global_add_pool(target_emb, g.batch, g.num_graphs)
        class_emb = class_emb / n_count.view(-1, 1)
        rep_class_emb = class_emb.repeat_interleave(g.num_classes, dim=0)
        res = self.mlp(
            torch.cat([rep_class_emb, g.x[g.true_nodes_mask]], dim=-1)
        )
        return res


class SingleHeadAtt(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sqrt_dim = torch.sqrt(torch.tensor(dim))
        self.Wk = torch.nn.Parameter(torch.zeros((dim, dim)))
        torch.nn.init.xavier_uniform_(self.Wk)
        self.Wq = torch.nn.Parameter(torch.zeros((dim, dim)))
        torch.nn.init.xavier_uniform_(self.Wq)

    def forward(self, key, query, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = torch.nn.functional.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class BinGraphModel(torch.nn.Module):
    def __init__(self, model, indim, outdim, task_dim, add_rwpe=None, dropout=0.0):
        super().__init__()
        self.model = model
        self.in_proj = nn.Linear(indim, outdim)
        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=dropout)
        if add_rwpe is not None:
            self.rwpe = AddRandomWalkPE(add_rwpe)
            self.edge_rwpe_prior = torch.nn.Parameter(
                torch.zeros((1, add_rwpe))
            )
            torch.nn.init.xavier_uniform_(self.edge_rwpe_prior)
            self.rwpe_normalization = torch.nn.BatchNorm1d(add_rwpe)
            self.walk_length = add_rwpe
        else:
            self.rwpe = None

    def initial_projection(self, g):
        g.x = self.in_proj(g.x)
        g.edge_attr = self.in_proj(g.edge_attr)
        return g

    def forward(self, g):
        g = self.initial_projection(g)

        if self.rwpe is not None:
            with torch.no_grad():
                rwpe_norm = self.rwpe_normalization(g.rwpe)
                g.x = torch.cat([g.x, rwpe_norm], dim=-1)
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        emb = self.model(g, g.feat_node_mask)
        class_emb = emb[g.true_nodes_mask]
        # print(class_emb)
        res = self.mlp(class_emb)
        return res


class BinGraphAttModel(torch.nn.Module):
    """
    GNN model that use a single layer attention to pool final node representation across
    layers.
    """
    def __init__(self, model, indim, outdim, task_dim, add_rwpe=None, dropout=0.0):
        super().__init__()
        self.model = model
        self.in_proj = nn.Linear(indim, outdim)

        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=dropout)
        self.att = SingleHeadAtt(outdim)
        if add_rwpe is not None:
            self.rwpe = AddRandomWalkPE(add_rwpe)
            self.edge_rwpe_prior = torch.nn.Parameter(
                torch.zeros((1, add_rwpe))
            )
            torch.nn.init.xavier_uniform_(self.edge_rwpe_prior)
            self.rwpe_normalization = torch.nn.BatchNorm1d(add_rwpe)
            self.walk_length = add_rwpe
        else:
            self.rwpe = None

    def initial_projection(self, g):
        g.x = self.in_proj(g.x)
        g.edge_attr = self.in_proj(g.edge_attr)
        return g

    def forward(self, g):
        g = self.initial_projection(g)
        if self.rwpe is not None:
            with torch.no_grad():
                rwpe_norm = self.rwpe_normalization(g.rwpe)
                g.x = torch.cat([g.x, rwpe_norm], dim=-1)
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        emb = torch.stack(self.model(g, g.feat_node_mask), dim=1)
        query = g.x.unsqueeze(1)
        emb = self.att(emb, query, emb)[0].squeeze()

        class_emb = emb[g.true_nodes_mask]
        # print(class_emb)
        res = self.mlp(class_emb)
        return res


class TransformerModel(nn.Module):
    """Transformer encoder model using Pytorch.
    Args:
        input_dim (int): Input dimension of the model.
        num_layers (int): Number of transformer layer.
        hidden_dim (int): Hidden dimension in transformer model.
        num_heads (int): Number of head in each transformer layer.

    """

    def __init__(
        self, input_dim: int, num_layers: int, hidden_dim: int, num_heads: int
    ):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                input_dim, num_heads, hidden_dim, batch_first=True
            ),
            num_layers,
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
        src_key_padding_mask: Tensor = None,
    ) -> Tensor:
        encoded = self.encoder(
            x, mask=mask, src_key_padding_mask=src_key_padding_mask
        )
        return encoded


class PyGRGCNEdge(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers: int,
        num_rels: int,
        inp_dim: int,
        out_dim: int,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels
        self.build_layers()

    def build_input_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_hidden_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_message_from_input(self, g):
        return {
            "g": g.edge_index,
            "h": g.x,
            "e": g.edge_type,
            "he": g.edge_attr,
        }

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h, "e": g.edge_type, "he": g.edge_attr}

    def layer_forward(self, layer, message):
        return self.conv[layer](
            message["h"], message["he"], message["g"], message["e"]
        )
