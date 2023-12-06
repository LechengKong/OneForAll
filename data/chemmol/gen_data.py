import os
import torch
import torch_geometric as pyg
import numpy as np
import json
from data.ofa_data import OFAPygDataset

import pickle
from datasets import load_dataset
import pandas as pd
from data.chemmol.gen_raw_graph import smiles2graph
from utils import get_label_texts

NAME_TO_SPLIT = {"chemblpre": "chembl_pretraining", "chempcba": "pcba", "chemhiv": "hiv"}


def load_prompt_json(name):
    if name == "chemblpre":
        with open(os.path.join(os.path.dirname(__file__), "prompt_pretrain.json"), "rb") as f:
            prompt_text = json.load(f)
        return prompt_text["chembl"]
    else:
        with open(os.path.join(os.path.dirname(__file__), "mol_label_desc.json"), "rb") as f:
            prompt_text = json.load(f)
        if name in NAME_TO_SPLIT:
            return prompt_text[NAME_TO_SPLIT[name]]
        else:
            raise NotImplementedError("Molecule dataset " + name + " not implemented.")


def get_local_text(name):
    print("gen text")
    cache_dir = os.path.join(os.path.dirname(__file__), "../../cache_data/dataset")
    data = load_dataset("haitengzhao/molecule_property_instruction", cache_dir=cache_dir, split=NAME_TO_SPLIT[name], )
    data_dict = {"label": data["label"], "task_index": data["task_index"], "molecule_index": data["molecule_index"], }
    pd_data = pd.DataFrame.from_dict(data_dict)
    cls_data = pd_data[np.logical_not(pd.isna(pd_data["task_index"]))]
    cls_data["ori_index"] = np.arange(len(cls_data))

    group = cls_data.groupby("molecule_index")
    index = group.ori_index.first()
    tasks = group.task_index.agg(lambda x: x.str.cat(sep=","))
    labels = group.label.agg(lambda x: x.str.cat(sep=","))
    mol = [data[i]["graph"] for i in index]
    split = [data[i]["split"] for i in index]

    prompt_text = load_prompt_json(name)
    task2index = {k: [i, prompt_text[k]] for i, k in enumerate(prompt_text)}
    label_text = get_label_texts(task2index)
    graphs = []
    for i in range(len(mol)):
        graph = smiles2graph(mol[i])
        task_lst = [task2index[v][0] for v in tasks[i].split(",")]
        label_lst = [1 if v == "Yes" else 0 for v in labels[i].split(",")]
        cur_label = np.zeros(len(task2index))
        cur_label[:] = np.nan
        cur_label[task_lst] = label_lst
        graph["label"] = cur_label
        graph["split"] = split[i]
        graphs.append(graph)
    return graphs, label_text


def gen_graph(graphs, labels_features):
    print("gen graph")

    node_texts = []
    edge_texts = []
    data = []
    for g in graphs:
        node_texts += g["node_feat"]
        edge_texts += g["edge_feat"]
    unique_node_texts = set(node_texts)
    unique_edge_texts = set(edge_texts)
    u_node_texts_lst = list(unique_node_texts)
    u_edge_texts_lst = list(unique_edge_texts)
    node_texts2id = {v: i for i, v in enumerate(u_node_texts_lst)}
    edge_texts2id = {v: i for i, v in enumerate(u_edge_texts_lst)}
    split = {"train": [], "valid": [], "test": []}
    for i, g in enumerate(graphs):
        cur_nt_id = [node_texts2id[v] for v in g["node_feat"]]
        cur_et_id = [edge_texts2id[v] for v in g["edge_feat"]]
        data.append(pyg.data.data.Data(x=torch.tensor(cur_nt_id, dtype=torch.long),
            xe=torch.tensor(cur_et_id, dtype=torch.long), edge_index=torch.tensor(g["edge_list"], dtype=torch.long).T,
            y=torch.tensor(g["label"]), ))
        split[g["split"]].append(i)

    prompt_edge_text = ["prompt edge.", "prompt edge. edge for query graph that is our target",
        "prompt edge. edge for support graph that is an example", ]
    prompt_text = ["prompt node. graph classification on molecule property",
        "prompt node. few shot task node for graph classification that decides whether the query molecule belongs to "
        "the class of support molecules.", ]

    prompt_text_map = {"e2e_graph": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                     "class_node_text_feat": ["class_node_text_feat",
                                                              torch.arange(len(labels_features))],
                                     "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
                       "lr_graph": {"noi_node_text_feat": ["noi_node_text_feat", [1]],
                                    "class_node_text_feat": ["class_node_text_feat",
                                                             torch.arange(len(labels_features))],
                                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]]}}

    ret = (data, [u_node_texts_lst, u_edge_texts_lst, labels_features, prompt_edge_text, prompt_text, ],
           [split, prompt_text_map],)
    return ret


class MolOFADataset(OFAPygDataset):
    def gen_data(self):
        pyg_graph, texts, split = gen_graph(*get_local_text(self.name))
        return [d for d in pyg_graph], texts, split

    def add_text_emb(self, data_list, text_emb):
        """
        Since the majority of node/edge text embeddings are repeated, we only store unique
        ones, and keep the indices.
        """
        data, slices = self.collate(data_list)
        data.node_embs = text_emb[0]
        data.edge_embs = text_emb[1]
        data.class_node_text_feat = text_emb[2]
        data.prompt_edge_text_feat = text_emb[3]
        data.noi_node_text_feat = text_emb[4]
        return data, slices

    def get(self, index):
        data = super().get(index)
        node_feat = self.node_embs[data.x]
        edge_feat = self.edge_embs[data.xe]
        data.node_text_feat = node_feat
        data.edge_text_feat = edge_feat
        data.y = data.y.view(1, -1)
        return data

    def get_idx_split(self):
        return self.side_data[0]

    def get_task_map(self):
        return self.side_data[1]

    def get_edge_list(self, mode="e2e"):
        if mode == "e2e_graph":
            return {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0]}
        elif mode == "lr_graph":
            return {"f2n": [1, 0], "n2f": [3, 0]}


if __name__ == "__main__":
    g, label = gen_graph()
    print(g[0])
