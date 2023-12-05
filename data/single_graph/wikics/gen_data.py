import json
import os
import functools
import torch
from data.ofa_data import OFAPygDataset
from torch_geometric.datasets import WikiCS


# For debug
#  from data.wikics.gen_data import get_text
#  a,b = get_text('wikics')


def get_text(path):
    """
    Returns: node_text_lst, label_text_lst
    Node text format: "wikipedia entry name: xxx. entry content: xxxxx"
    Label text format: "wikipedia entry category: xxx"
    """
    with open(os.path.join(path, "metadata.json")) as json_file:
        raw_data = json.load(json_file)

    node_info = raw_data["nodes"]
    label_info = raw_data["labels"]
    node_text_lst = []
    label_text_lst = []

    # Process Node Feature
    for node in node_info:
        node_feature = ((
                "feature node. wikipedia entry name: " + node["title"] + ". entry content: " + functools.reduce(
            lambda x, y: x + " " + y, node["tokens"])).lower().strip())
        node_text_lst.append(node_feature)

    # Process Label Feature
    for label in label_info.values():
        label_feature = (("prompt node. wikipedia entry category: " + label).lower().strip())
        label_text_lst.append(label_feature)

    return node_text_lst, label_text_lst


def get_data(dset):
    pyg_data = WikiCS(root=dset.data_dir)
    cur_path = os.path.dirname(__file__)
    node_texts, label_texts = get_text(cur_path)
    edge_text = ["feature edge. wikipedia page link"]
    prompt_text = ["prompt node. node classification of wikipedia entry category"]
    prompt_edge_text = ["prompt edge."]
    prompt_text_map = {"e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                    "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_texts))],
                                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]}}
    return ([pyg_data.data], [node_texts, edge_text, prompt_text, label_texts, prompt_edge_text, ], prompt_text_map,)
