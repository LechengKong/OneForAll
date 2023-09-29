import os
import pandas as pd
import torch
import torch_geometric as pyg
from data.ofa_data import OFAPygDataset
from ogb.nodeproppred import PygNodePropPredDataset


def get_data(path):
    data = torch.load(path)
    print(data)
    text = data.raw_texts
    nx_g = pyg.utils.to_networkx(data, to_undirected=True)
    edge_index = torch.tensor(list(nx_g.edges())).T
    print(edge_index.size())
    data_dict = data.to_dict()
    data_dict["edge_index"] = edge_index
    new_data = pyg.data.data.Data(**data_dict)
    with open(
        os.path.join(os.path.dirname(__file__), "categories.csv"), "r"
    ) as f:
        ordered_desc = f.read().split("\n")
    clean_text = ["feature node. paper title and abstract: " + t for t in text]
    label_text = [
        "prompt node. literature category and description: " + desc
        for desc in ordered_desc
    ]
    edge_label_text = [
        "prompt node. two papers have co-citation",
        "prompt node. two papers do not have co-citation",
    ]
    edge_text = [
        "feature edge. connected papers are cited together by other papers."
    ]
    prompt_node_text = [
        "prompt node. link prediction on the papers that are cited together"
    ]
    prompt_node_edge_text = [
        "prompt node. node classification on the paper's category"
    ]
    prompt_edge_text = ["prompt edge."]
    return (
        [new_data],
        [
            clean_text,
            label_text,
            edge_text,
            prompt_node_text,
            prompt_node_edge_text,
            prompt_edge_text,
            edge_label_text,
        ],
        None,
    )


class PubmedOFADataset(OFAPygDataset):
    def gen_data(self):
        cur_path = os.path.dirname(__file__)
        return get_data(os.path.join(cur_path, "pubmed.pt"))

    def add_text_emb(self, data_list, text_emb):
        data_list[0].x_text_feat = text_emb[0]
        data_list[0].label_text_feat = text_emb[1]
        data_list[0].edge_text_feat = text_emb[2]
        data_list[0].prompt_node_feat = text_emb[3]
        data_list[0].prompt_node_edge_feat = text_emb[4]
        data_list[0].prompt_edge_feat = text_emb[5]
        data_list[0].edge_label_feat = text_emb[6]
        return self.collate(data_list)
