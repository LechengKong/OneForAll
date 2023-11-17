import os
import pandas as pd
import torch
from data.ofa_data import OFAPygDataset
from ogb.nodeproppred import PygNodePropPredDataset


def get_node_feature(path):
    # Node feature process
    nodeidx2paperid = pd.read_csv(
        os.path.join(path, "nodeidx2paperid.csv.gz"), index_col="node idx"
    )
    titleabs_url = (
        "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv"
    )
    titleabs = pd.read_csv(
        titleabs_url,
        sep="\t",
        names=["paper id", "title", "abstract"],
        index_col="paper id",
    )

    titleabs = nodeidx2paperid.join(titleabs, on="paper id")
    text = (
            "feature node. paper title and abstract: "
            + titleabs["title"]
            + ". "
            + titleabs["abstract"]
    )
    node_text_lst = text.values

    return node_text_lst


def get_taxonomy(path):
    # read categories and description file
    f = open(os.path.join(path, "arxiv_CS_categories.txt"), "r").readlines()

    state = 0
    result = {"id": [], "name": [], "description": []}

    for line in f:
        if state == 0:
            assert line.strip().startswith("cs.")
            category = (
                    "arxiv "
                    + " ".join(line.strip().split(" ")[0].split(".")).lower()
            )  # e. g. cs lo
            name = line.strip()[7:-1]  # e. g. Logic in CS
            result["id"].append(category)
            result["name"].append(name)
            state = 1
            continue
        elif state == 1:
            description = line.strip()
            result["description"].append(description)
            state = 2
            continue
        elif state == 2:
            state = 0
            continue

    arxiv_cs_taxonomy = pd.DataFrame(result)

    return arxiv_cs_taxonomy


def get_label_feature(path):
    arxiv_cs_taxonomy = get_taxonomy(path)
    mapping_file = os.path.join(path, "labelidx2arxivcategeory.csv.gz")
    arxiv_categ_vals = pd.merge(
        pd.read_csv(mapping_file),
        arxiv_cs_taxonomy,
        left_on="arxiv category",
        right_on="id",
    )

    text = (
            "prompt node. literature category and description: "
            + arxiv_categ_vals["name"]
            + ". "
            + arxiv_categ_vals["description"]
    )
    label_text_lst = text.values

    return label_text_lst


def get_data(dset):
    pyg_data = PygNodePropPredDataset(
        name="ogbn-arxiv", root=dset.data_dir
    )
    cur_path = os.path.dirname(__file__)
    feat_node_texts = get_node_feature(cur_path).tolist()
    class_node_texts = get_label_feature(cur_path).tolist()
    feat_edge_texts = ["feature edge. citation"]
    noi_node_texts = [
        "prompt node. node classification of literature category"
    ]
    prompt_edge_texts = [
        "prompt edge.",
        "prompt edge. edge for query graph that is our target",
        "prompt edge. edge for support graph that is an example",
    ]
    prompt_text_map = {"e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                    "class_node_text_feat": ["class_node_text_feat",
                                                             torch.arange(len(class_node_texts))],
                                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
                       "lr_e2e": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                   "class_node_text_feat": ["class_node_text_feat",
                                                            torch.arange(len(class_node_texts))],
                                   "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]]}}
    return (
        [pyg_data.data],
        [
            feat_node_texts,
            feat_edge_texts,
            noi_node_texts,
            class_node_texts,
            prompt_edge_texts,
        ],
        prompt_text_map,
    )
