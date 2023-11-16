"""Base message-passing GNNs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from gp.nn.models.util_model import MLP

from abc import ABCMeta, abstractmethod

from gp.utils.utils import SmartTimer

from torch_scatter import scatter


class MultiLayerMessagePassing(nn.Module, metaclass=ABCMeta):
    """Message passing GNN"""

    def __init__(
            self,
            num_layers,
            inp_dim,
            out_dim,
            drop_ratio=None,
            JK="last",
            batch_norm=True,
    ):
        """

        :param num_layers: layer number of GNN
        :type num_layers: int
        :param inp_dim: input feature dimension
        :type inp_dim: int
        :param out_dim: output dimension
        :type out_dim: int
        :param drop_ratio: layer-wise node dropout ratio, defaults to None
        :type drop_ratio: float, optional
        :param JK: jumping knowledge, should either be ["last","sum"],
        defaults to "last"
        :type JK: str, optional
        :param batch_norm: Use node embedding batch normalization, defaults
        to True
        :type batch_norm: bool, optional
        """
        super().__init__()
        self.num_layers = num_layers
        self.JK = JK
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        self.drop_ratio = drop_ratio

        self.conv = torch.nn.ModuleList()

        if batch_norm:
            self.batch_norm = torch.nn.ModuleList()
            for layer in range(num_layers):
                self.batch_norm.append(torch.nn.BatchNorm1d(out_dim))
        else:
            self.batch_norm = None

        self.timer = SmartTimer(False)

    def build_layers(self):
        for layer in range(self.num_layers):
            if layer == 0:
                self.conv.append(self.build_input_layer())
            else:
                self.conv.append(self.build_hidden_layer())

    @abstractmethod
    def build_input_layer(self):
        pass

    @abstractmethod
    def build_hidden_layer(self):
        pass

    @abstractmethod
    def layer_forward(self, layer, message):
        pass

    @abstractmethod
    def build_message_from_input(self, g):
        pass

    @abstractmethod
    def build_message_from_output(self, g, output):
        pass

    def forward(self, g, drop_mask=None):
        h_list = []

        message = self.build_message_from_input(g)

        for layer in range(self.num_layers):
            # print(layer, h)
            h = self.layer_forward(layer, message)
            if self.batch_norm:
                h = self.batch_norm[layer](h)
            if layer != self.num_layers - 1:
                h = F.relu(h)
            if self.drop_ratio is not None:
                dropped_h = F.dropout(h, p=self.drop_ratio, training=self.training)
                if drop_mask is not None:
                    h = drop_mask.view(-1, 1) * dropped_h + torch.logical_not(drop_mask).view(-1, 1) * h
                else:
                    h = dropped_h
            message = self.build_message_from_output(g, h)
            h_list.append(h)

        if self.JK == "last":
            repr = h_list[-1]
        elif self.JK == "sum":
            repr = 0
            for layer in range(self.num_layers):
                repr += h_list[layer]
        elif self.JK == "mean":
            repr = 0
            for layer in range(self.num_layers):
                repr += h_list[layer]
            repr = repr/self.num_layers
        else:
            repr = h_list
        return repr


class MultiLayerMessagePassingVN(MultiLayerMessagePassing):
    def __init__(
            self,
            num_layers,
            inp_dim,
            out_dim,
            drop_ratio=None,
            JK="last",
            batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )

        self.virtualnode_embedding = torch.nn.Embedding(1, self.out_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.virtualnode_mlp_list = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            self.virtualnode_mlp_list.append(
                MLP([self.out_dim, 2 * self.out_dim, self.out_dim])
            )

    def forward(self, g):
        h_list = []

        message = self.build_message_from_input(g)

        vnode_embed = self.virtualnode_embedding(
            torch.zeros(g.batch_size, dtype=torch.int).to(g.device)
        )

        batch_node_segment = torch.arange(
            g.batch_size, dtype=torch.long, device=g.device
        ).repeat_interleave(g.batch_num_nodes())

        for layer in range(self.num_layers):
            # print(layer, h)
            h = self.layer_forward(layer, message)
            if self.batch_norm:
                h = self.batch_norm[layer](h)
            if layer != self.num_layers - 1:
                h = F.relu(h)
            if self.drop_ratio is not None:
                h = F.dropout(h, p=self.drop_ratio, training=self.training)
            message = self.build_message_from_output(g, h)
            h_list.append(h)

            if layer < self.num_layers - 1:
                vnode_emb_temp = (
                        scatter(
                            h, batch_node_segment, dim=0, dim_size=g.batch_size
                        )
                        + vnode_embed
                )

                vnode_embed = F.dropout(
                    self.virtualnode_mlp_list[layer](vnode_emb_temp),
                    self.drop_ratio,
                    training=self.training,
                )

        if self.JK == "last":
            repr = h_list[-1]
        elif self.JK == "sum":
            repr = 0
            for layer in range(self.num_layers):
                repr += h_list[layer]
        elif self.JK == "cat":
            repr = torch.cat([h_list], dim=-1)
        return repr
