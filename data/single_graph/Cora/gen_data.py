import os
import pandas as pd
import torch
import torch_geometric as pyg
from data.ofa_data import OFAPygDataset
from ogb.nodeproppred import PygNodePropPredDataset


def get_logic_label(ordered_txt):
    or_labeled_text = []
    not_and_labeled_text = []
    for i in range(len(ordered_txt)):
        for j in range(len(ordered_txt)):
            c1 = ordered_txt[i]
            c2 = ordered_txt[j]
            txt = "prompt node. literature category and description: not " + c1[0] + ". " + c1[1][0] + " and not " + c2[
                0] + ". " + c2[1][0]
            not_and_labeled_text.append(txt)
            txt = "prompt node. literature category and description: either " + c1[0] + ". " + c1[1][0] + " or " + c2[
                0] + ". " + c2[1][0]
            or_labeled_text.append(txt)
    return or_labeled_text + not_and_labeled_text


def get_data(dset):
    cur_path = os.path.dirname(__file__)
    path = os.path.join(cur_path, "cora.pt")
    data = torch.load(path)
    text = data.raw_texts
    label_names = data.label_names
    nx_g = pyg.utils.to_networkx(data, to_undirected=True)
    edge_index = torch.tensor(list(nx_g.edges())).T
    print(edge_index.size())
    data_dict = data.to_dict()
    data_dict["edge_index"] = edge_index
    new_data = pyg.data.data.Data(**data_dict)
    category_desc = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "categories.csv"), sep=","
    ).values
    ordered_desc = []
    for i, label in enumerate(label_names):
        true_ind = label == category_desc[:, 0]
        ordered_desc.append((label, category_desc[:, 1][true_ind]))
    clean_text = ["feature node. paper title and abstract: " + t for t in text]
    label_text = [
        "prompt node. literature category and description: "
        + desc[0]
        + "."
        + desc[1][0]
        for desc in ordered_desc
    ]
    edge_label_text = [
        "prompt node. two papers do not have co-citation",
        "prompt node. two papers have co-citation"
    ]
    logic_label_text = get_logic_label(ordered_desc)
    edge_text = [
        "feature edge. connected papers are cited together by other papers."
    ]
    noi_node_edge_text = [
        "prompt node. link prediction on the papers that are cited together"
    ]
    noi_node_text = [
        "prompt node. node classification on the paper's category"
    ]
    prompt_edge_text = ["prompt edge", "prompt edge. edge for query graph that is our target",
                        "prompt edge. edge for support graph that is an example"]
    return (
        [new_data],
        [
            clean_text,
            edge_text,
            noi_node_text + noi_node_edge_text,
            label_text + edge_label_text + logic_label_text,
            prompt_edge_text,
        ],
        {"e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                      "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_text))],
                      "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
         "e2e_link": {"noi_node_text_feat": ["noi_node_text_feat", [1]],
                      "class_node_text_feat": ["class_node_text_feat",
                                               torch.arange(len(label_text), len(label_text) + len(edge_label_text))],
                      "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
         "lr_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                     "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_text))],
                     "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]]},
         "logic_e2e": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                       "class_node_text_feat": ["class_node_text_feat",
                                                torch.arange(len(label_text) + len(edge_label_text),
                                                             len(label_text) + len(edge_label_text) + len(
                                                                 logic_label_text))],
                       "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
         }
    )
