import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax, add_self_loops


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, torch.Tensor):
        return edge_index[:, edge_mask]


class RGCNEdgeConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        aggr: str = "mean",
        **kwargs,
    ):
        kwargs.setdefault("aggr", aggr)
        super().__init__(**kwargs)  # "Add" aggregation (Step 5).
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.weight = Parameter(
            torch.empty(self.num_relations, in_channels, out_channels)
        )

        self.root = Parameter(torch.empty(in_channels, out_channels))
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.root)
        zeros(self.bias)

    def forward(
        self,
        x: OptTensor,
        xe: OptTensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
    ):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        for i in range(self.num_relations):
            edge_mask = edge_type == i
            tmp = masked_edge_index(edge_index, edge_mask)

            h = self.propagate(tmp, x=x, xe=xe[edge_mask])
            out += h @ self.weight[i]

        out += x @ self.root
        out += self.bias

        return out

    def message(self, x_j, xe):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return (x_j + xe).relu()


class RGATEdgeConv(RGCNEdgeConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        aggr: str = "sum",
        heads=8,
        add_self_loops=False,
        share_att=False,
        **kwargs,
    ):
        self.heads = heads
        self.add_self_loops = add_self_loops
        self.share_att = share_att
        super().__init__(
            in_channels,
            out_channels,
            num_relations,
            aggr,
            node_dim=0,
            **kwargs,
        )
        self.lin_edge = nn.Linear(self.in_channels, self.out_channels)
        assert self.in_channels % heads == 0
        self.d_model = self.in_channels // heads
        if self.share_att:
            self.att = Parameter(torch.empty(1, heads, self.d_model))
        else:
            self.att = Parameter(
                torch.empty(self.num_relations, heads, self.d_model)
            )

        glorot(self.att)

    def forward(
        self,
        x: OptTensor,
        xe: OptTensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
    ):
        out = torch.zeros((x.size(0), self.out_channels), device=x.device)

        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, xe = add_self_loops(
                edge_index, xe, fill_value="mean", num_nodes=num_nodes
            )

        x_ = x.view(-1, self.heads, self.d_model)
        xe_ = self.lin_edge(xe).view(-1, self.heads, self.d_model)

        for i in range(self.num_relations):
            edge_mask = edge_type == i
            if self.add_self_loops:
                edge_mask = torch.cat(
                    [
                        edge_mask,
                        torch.ones(num_nodes, device=edge_mask.device).bool(),
                    ]
                )

            tmp = masked_edge_index(edge_index, edge_mask)

            h = self.propagate(tmp, x=x_, xe=xe_[edge_mask], rel_index=i)
            h = h.view(-1, self.in_channels)
            out += h @ self.weight[i]

        out += x @ self.root
        out += self.bias

        return out

    def message(self, x_j, xe, rel_index, index, ptr, size_i):
        # x_j has shape [E, out_channels]
        x = F.leaky_relu(x_j + xe)
        if self.share_att:
            att = self.att

        else:
            att = self.att[rel_index : rel_index + 1]

        alpha = (x * att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)

        return (x_j + xe) * alpha.unsqueeze(-1)
