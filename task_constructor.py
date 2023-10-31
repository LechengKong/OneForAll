import torch
import torch_geometric as pyg
import json

from data.arxiv.gen_data import ArxivOFADataset
from data.Cora.gen_data import CoraOFADataset
from data.Pubmed.gen_data import PubmedOFADataset
from data.WN18RR.gen_data import WN18RROFADataset
from data.FB15K237.gen_data import FB15K237OFADataset
from data.wikics.gen_data import WikiCSOFADataset
from data.chemblpre.gen_data import CHEMBLPREOFADataset
from data.chempcba.gen_data import CHEMPCBAOFADataset
from data.chemhiv.gen_data import CHEMHIVOFADataset

from ofa_datasets import (
    GraphListDataset,
    SubgraphDataset,
    MultiDataset,
    GraphListHierDataset,
    SubgraphHierDataset,
    GraphListHierFSDataset,
    GraphListHierFixDataset,
    SubgraphLinkHierDataset,
    SubgraphKGHierDataset,
    FewShotNCDataset,
    FewShotKGDataset,
    ZeroShotNCDataset,
    ZeroShotKGDataset,
    SubgraphNopromptDataset,
    GraphListNopromptDataset,
)
from fs_datamanager import FewShotDataManager

from gp.utils.utils import k_fold_ind, k_fold2_split
from gp.lightning.data_template import DataWithMeta

from gp.lightning.metric import (
    binary_auc_func,
    flat_binary_func,
    classification_func,
    EvalKit,
)
from utils import (
    binary_apr_func,
    binary_auc_multi_func,
    binary_single_auc_func,
    classification_single_func,
)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

name2dataset = {
    "arxiv": ArxivOFADataset,
    "cora": CoraOFADataset,
    "pubmed": PubmedOFADataset,
    "WN18RR": WN18RROFADataset,
    "FB15K237": FB15K237OFADataset,
    "wikics": WikiCSOFADataset,
    "chemblpre": CHEMBLPREOFADataset,
    "chempcba": CHEMPCBAOFADataset,
    "chemhiv": CHEMHIVOFADataset,
}


def ArxivSplitter(dataset):
    text_g = dataset.data
    kfold = k_fold_ind(text_g.y, 10)
    text_split = k_fold2_split(kfold, len(text_g.y))[0]
    split = {}
    split["train"] = text_split[0]
    split["valid"] = text_split[1]
    split["test"] = text_split[2]
    return split


def CiteSplitter(dataset):
    text_g = dataset.data
    split = {
        "train": text_g.train_masks[0].nonzero(as_tuple=True)[0],
        "valid": text_g.val_masks[0].nonzero(as_tuple=True)[0],
        "test": text_g.test_masks[0].nonzero(as_tuple=True)[0],
    }
    return split


def CiteLinkSplitter(dataset):
    text_g = dataset.data
    text_g.x = text_g.x_text_feat
    text_g.prompt_edge_feat = dataset.prompt_edge_feat
    edges = text_g.edge_index
    edge_perm = torch.randperm(len(edges[0]))
    train_offset = int(len(edge_perm) * 0.85)
    val_offset = int(len(edge_perm) * 0.9)
    edge_indices = {
        "train": edge_perm[:train_offset],
        "valid": edge_perm[train_offset:val_offset],
        "test": edge_perm[val_offset:],
    }
    return edge_indices


def KGSplitter(dataset):
    converted_triplet = dataset.get_idx_split()
    return converted_triplet


def WikiSplitter(dataset):
    text_g = dataset.data
    wiki_split_idx = 0
    split = {
        "train": torch.where(text_g.train_mask[:, wiki_split_idx])[0].numpy(),
        "valid": torch.where(text_g.val_mask[:, wiki_split_idx])[0].numpy(),
        "test": torch.where(text_g.test_mask)[0].numpy(),
    }
    return split


def MolSplitter(dataset):
    return dataset.get_idx_split()


name2splitter = {
    "arxiv": ArxivSplitter,
    "cora_node": CiteSplitter,
    "pubmed_node": CiteSplitter,
    "cora_link": CiteLinkSplitter,
    "pubmed_link": CiteLinkSplitter,
    "WN18RR": KGSplitter,
    "FB15K237": KGSplitter,
    "wikics": WikiSplitter,
    "chemblpre": MolSplitter,
    "chempcba": MolSplitter,
    "chemhiv": MolSplitter,
}


def LinkConstructGraph(dataset, split):
    text_g = dataset.data
    edges = text_g.edge_index
    graph_dict = text_g.to_dict()
    graph_dict["edge_index"] = edges[:, split["train"]]
    train_graph = pyg.data.Data(**graph_dict)
    return train_graph


def make_data(
    name, data, split_name, metric, eval_func, num_classes, **kwargs
):
    return DataWithMeta(
        data,
        kwargs["batch_size"],
        sample_size=kwargs["sample_size"],
        metric=metric,
        state_name=split_name + "_" + name,
        classes=num_classes,
        meta_data={"eval_func": eval_func},
    )


def ConstructNodeCls(
    name, dataset, split, split_name, to_bin_cls_func, **kwargs
):
    text_g = dataset.data

    return SubgraphHierDataset(
        text_g,
        text_g.label_text_feat,
        split[split_name],
        prompt_feat=text_g.prompt_text_feat,
        to_undirected=True,
        process_label_func=to_bin_cls_func,
        walk_length=kwargs["walk_length"],
    )


def ConstructLinkCls(
    name, dataset, split, split_name, to_bin_cls_func, **kwargs
):
    text_g = dataset.data
    edges = text_g.edge_index
    train_graph = kwargs["global_data"]

    return SubgraphLinkHierDataset(
        train_graph,
        train_graph.edge_label_feat,
        edges.T[split[split_name]].numpy(),
        prompt_feat=train_graph.prompt_text_edge_feat,
        to_undirected=True,
        hop=3,
        remove_edge=kwargs["remove_edge"],
        process_label_func=to_bin_cls_func,
        walk_length=kwargs["walk_length"],
    )


def ConstructKG(name, dataset, split, split_name, to_bin_cls_func, **kwargs):
    text_g = dataset.data

    return SubgraphKGHierDataset(
        text_g,
        text_g.edge_label_feat,
        split[split_name],
        prompt_feat=text_g.prompt_text_feat,
        to_undirected=True,
        hop=2,
        remove_edge=kwargs["remove_edge"],
        process_label_func=to_bin_cls_func,
        walk_length=kwargs["walk_length"],
    )


def ConstructMolCls(
    name, dataset, split, split_name, to_bin_cls_func, **kwargs
):

    return GraphListHierDataset(
        dataset,
        dataset.label_text_feat,
        dataset.prompt_edge_feat,
        dataset.prompt_text_feat,
        split[split_name],
        process_label_func=to_bin_cls_func,
        single_prompt_edge=True,
        walk_length=kwargs["walk_length"],
    )


def ConstructMolFSTrain(
    name, dataset, split, split_name, to_bin_cls_func, **kwargs
):
    classes = dataset.y.view(len(dataset), -1)[split["train"]]

    return GraphListHierFSDataset(
        dataset,
        dataset.label_text_feat,
        dataset.prompt_edge_feat,
        dataset.prompt_text_feat,
        split[split_name],
        process_label_func=to_bin_cls_func,
        single_prompt_edge=True,
        walk_length=kwargs["walk_length"],
        class_ind=classes,
    )

def ConstructNCFSZS(
        dataset, data_manager, n, k, split_name, random_flag=False, state_name=None, eval_metric=None, eval_func=None, train_flag=False, adj=None, total_task_num=50, class_emb_flag=False, undirected_flag=True,  **kwargs
):
    if class_emb_flag:
        class_emb = dataset.label_text_feat
    else:
        class_emb = dataset.prompt_text_feat.repeat(
            len(dataset.label_text_feat), 1
        )
    data_class = FewShotNCDataset if kwargs["k_shot"]>0 else ZeroShotNCDataset
    ofa_data = data_class(
        pyg_graph=dataset,
        class_emb=class_emb,
        data_idx=torch.zeros(total_task_num),
        n_way=kwargs["n_way"],
        k_shot=kwargs["k_shot"],
        q_query=kwargs["q_query"],
        datamanager=data_manager,
        mode=split_name,
        hop=2,
        prompt_feat=dataset.prompt_text_feat,
        to_undirected=undirected_flag,
        adj=adj,
        single_prompt_edge=True,
        random_flag=random_flag,
        min_n=n,
        min_k=k,
    )

    if train_flag:
        return ofa_data
    else:
        return DataWithMeta(
            ofa_data,
            batch_size=kwargs["fs_task_num"],
            sample_size=-1,
            metric=eval_metric,
            state_name=state_name,
            classes=n,
            meta_data={"eval_func": eval_func}
        )

def ConstructLPFSZS(
        dataset, data_manager, n, k, split_name, random_flag=False, state_name=None, eval_metric=None, eval_func=None, train_flag=False, adj=None, total_task_num=50, class_emb_flag=False, undirected_flag=True,  **kwargs
):
    if class_emb_flag:
        class_emb = dataset.edge_label_feat
    else:
        class_emb = dataset.prompt_text_feat.repeat(
            len(dataset.edge_label_feat), 1
        )
    data_class = FewShotKGDataset if kwargs["k_shot"]>0 else ZeroShotKGDataset
    ofa_data = data_class(
        pyg_graph=dataset,
        class_emb=class_emb,
        data_idx=torch.zeros(total_task_num),
        n_way=kwargs["n_way"],
        k_shot=kwargs["k_shot"],
        q_query=kwargs["q_query"],
        datamanager=data_manager,
        mode=split_name,
        edges=dataset.edge_index,
        fs_edges=kwargs["edges"]["fs_edges"][split_name],
        fs_edge_types=kwargs["edges"]["fs_edge_types"][split_name],
        hop=2,
        prompt_feat=dataset.prompt_text_feat,
        to_undirected=undirected_flag,
        adj=adj,
        single_prompt_edge=True,
        random_flag=random_flag,
        min_n=n,
        min_k=k,
    )

    if train_flag:
        return ofa_data
    else:
        return DataWithMeta(
            ofa_data,
            batch_size=kwargs["fs_task_num"],
            sample_size=-1,
            metric=eval_metric,
            state_name=state_name,
            classes=n,
            meta_data={"eval_func": eval_func}
        )



def process_pth_label(embs, label):
    binary_rep = torch.zeros((1, len(embs)))
    binary_rep[0, label] = 1
    return label.view(1, -1), embs, binary_rep


def process_multi_label(embs, label):
    valid_idx = label == label
    # valid_idx = torch.zeros_like(classes, dtype=torch.bool)
    return (
        torch.tensor([[0]]),
        embs[valid_idx.view(-1)].detach().clone(),
        label[:, valid_idx.view(-1)].detach().clone(),
    )


def process_int_label(embs, label):
    binary_rep = torch.zeros((1, len(embs)))
    binary_rep[0, label] = 1
    return torch.tensor([label]).view(1, -1), embs, binary_rep


none_process_label = None


task_config_lookup = {
    "arxiv": {
        "dataset_name": "arxiv",
        "dataset_splitter": "ArxivSplitter",
        "preprocess": None,
        "construct": "ConstructNodeCls",
        "args": {"walk_length": None},
        "process_label_func": "process_pth_label",
        "eval_set_constructs": [
            {
                "stage": "valid",
                "split_name": "valid",
            },
            {
                "stage": "test",
                "split_name": "test",
            },
            {
                "stage": "test",
                "split_name": "train",
            },
        ],
        "eval_metric": "acc",
        "eval_func": "classification_func",
        "num_classes": 40,
    },
    "cora_link": {
        "dataset_name": "cora",
        "dataset_splitter": "CiteLinkSplitter",
        "preprocess": "LinkConstructGraph",
        "construct": "ConstructLinkCls",
        "args": {"remove_edge": True, "walk_length": None},
        "process_label_func": "process_int_label",
        "eval_set_constructs": [
            {
                "stage": "valid",
                "split_name": "valid",
                "args": {"remove_edge": False, "walk_length": None},
            },
            {
                "stage": "test",
                "split_name": "test",
                "args": {"remove_edge": False, "walk_length": None},
            },
            {
                "stage": "test",
                "split_name": "train",
                "args": {"remove_edge": True, "walk_length": None},
            },
        ],
        "eval_metric": "auc",
        "eval_func": "binary_auc_func",
        "num_classes": 2,
    },
    "cora_node": {
        "dataset_name": "cora",
        "dataset_splitter": "CiteSplitter",
        "preprocess": None,
        "construct": "ConstructNodeCls",
        "args": {"walk_length": None},
        "process_label_func": "process_int_label",
        "eval_set_constructs": [
            {
                "stage": "valid",
                "split_name": "valid",
            },
            {
                "stage": "test",
                "split_name": "test",
            },
            {
                "stage": "test",
                "split_name": "train",
            },
        ],
        "eval_metric": "acc",
        "eval_func": "classification_func",
        "num_classes": 7,
    },
    "pubmed_link": {
        "dataset_name": "pubmed",
        "dataset_splitter": "CiteLinkSplitter",
        "preprocess": "LinkConstructGraph",
        "construct": "ConstructLinkCls",
        "args": {"remove_edge": True, "walk_length": None},
        "process_label_func": "process_int_label",
        "eval_set_constructs": [
            {
                "stage": "valid",
                "split_name": "valid",
                "args": {"remove_edge": False, "walk_length": None},
            },
            {
                "stage": "test",
                "split_name": "test",
                "args": {"remove_edge": False, "walk_length": None},
            },
            {
                "stage": "test",
                "split_name": "train",
                "args": {"remove_edge": True, "walk_length": None},
            },
        ],
        "eval_metric": "auc",
        "eval_func": "binary_auc_func",
        "num_classes": 2,
    },
    "pubmed_node": {
        "dataset_name": "pubmed",
        "dataset_splitter": "CiteSplitter",
        "preprocess": None,
        "construct": "ConstructNodeCls",
        "args": {"walk_length": None},
        "process_label_func": "process_int_label",
        "eval_set_constructs": [
            {
                "stage": "valid",
                "split_name": "valid",
            },
            {
                "stage": "test",
                "split_name": "test",
            },
            {
                "stage": "test",
                "split_name": "train",
            },
        ],
        "eval_metric": "acc",
        "eval_func": "classification_func",
        "num_classes": 3,
    },
    "WN18RR": {
        "dataset_name": "WN18RR",
        "dataset_splitter": "KGSplitter",
        "preprocess": None,
        "construct": "ConstructKG",
        "args": {"remove_edge": True, "walk_length": None},
        "process_label_func": "process_int_label",
        "eval_set_constructs": [
            {
                "stage": "valid",
                "split_name": "valid",
                "args": {"remove_edge": False, "walk_length": None},
            },
            {
                "stage": "test",
                "split_name": "test",
                "args": {"remove_edge": False, "walk_length": None},
            },
            {
                "stage": "test",
                "split_name": "train",
                "args": {"remove_edge": True, "walk_length": None},
            },
        ],
        "eval_metric": "acc",
        "eval_func": "classification_func",
        "num_classes": 11,
    },
    "FB15K237": {
        "dataset_name": "FB15K237",
        "dataset_splitter": "KGSplitter",
        "preprocess": None,
        "construct": "ConstructKG",
        "args": {"remove_edge": True, "walk_length": None},
        "process_label_func": "process_int_label",
        "eval_set_constructs": [
            {
                "stage": "valid",
                "split_name": "valid",
                "args": {"remove_edge": False, "walk_length": None},
            },
            {
                "stage": "test",
                "split_name": "test",
                "args": {"remove_edge": False, "walk_length": None},
            },
            {
                "stage": "test",
                "split_name": "train",
                "args": {"remove_edge": True, "walk_length": None},
            },
        ],
        "eval_metric": "acc",
        "eval_func": "classification_func",
        "num_classes": 237,
    },
    "wikics": {
        "dataset_name": "wikics",
        "dataset_splitter": "WikiSplitter",
        "preprocess": None,
        "construct": "ConstructNodeCls",
        "args": {"walk_length": None},
        "process_label_func": "process_pth_label",
        "eval_set_constructs": [
            {
                "stage": "valid",
                "split_name": "valid",
            },
            {
                "stage": "test",
                "split_name": "test",
            },
            {
                "stage": "test",
                "split_name": "train",
            },
        ],
        "eval_metric": "acc",
        "eval_func": "classification_func",
        "num_classes": 10,
    },
    "chemblpre": {
        "dataset_name": "chemblpre",
        "dataset_splitter": "MolSplitter",
        "preprocess": None,
        "construct": "ConstructMolCls",
        "args": {"walk_length": None},
        "process_label_func": "process_multi_label",
        "eval_set_constructs": [
            {
                "stage": "valid",
                "split_name": "valid",
                "process_label_func": "none_process_label",
            },
            {
                "stage": "test",
                "split_name": "test",
                "process_label_func": "none_process_label",
            },
            {
                "stage": "test",
                "split_name": "train",
                "process_label_func": "none_process_label",
            },
        ],
        "eval_metric": "apr",
        "eval_func": "binary_apr_func",
        "num_classes": 1296,
    },
    "chempcba": {
        "dataset_name": "chempcba",
        "dataset_splitter": "MolSplitter",
        "preprocess": None,
        "construct": "ConstructMolCls",
        "args": {"walk_length": None},
        "process_label_func": "process_multi_label",
        "eval_set_constructs": [
            {
                "stage": "valid",
                "split_name": "valid",
                "process_label_func": "none_process_label",
            },
            {
                "stage": "test",
                "split_name": "test",
                "process_label_func": "none_process_label",
            },
            {
                "stage": "test",
                "split_name": "train",
                "process_label_func": "none_process_label",
            },
        ],
        "eval_metric": "apr",
        "eval_func": "binary_apr_func",
        "num_classes": 128,
    },
    "chemhiv": {
        "dataset_name": "chemhiv",
        "dataset_splitter": "MolSplitter",
        "preprocess": None,
        "construct": "ConstructMolCls",
        "args": {"walk_length": None},
        "process_label_func": "process_pth_label",
        "eval_set_constructs": [
            {
                "stage": "valid",
                "split_name": "valid",
                "process_label_func": "none_process_label",
            },
            {
                "stage": "test",
                "split_name": "test",
                "process_label_func": "none_process_label",
            },
            {
                "stage": "test",
                "split_name": "train",
                "process_label_func": "none_process_label",
            },
        ],
        "eval_metric": "auc",
        "eval_func": "binary_auc_func",
        "num_classes": 2,
    },
    "arxiv_fs": {
            "dataset_name": "arxiv",
            "task_level": "node",
            "construct": "ConstructNCFSZS",
            "args": {"walk_length": None, "n_way": 5, "min_n": 3, "val_n": [5, 3], "k_shot": 5, "min_k": 1, "val_k": [1, 3, 5], "q_query": 1, "fs_task_num": 1, "class_split_ratio": None, },
            "mode": {"train": 0, "valid":1, "test":2},
            "eval_set_constructs": [
                {
                    "stage": "valid",
                },
                {
                    "stage": "test",
                },
            ],
            "eval_metric": "acc",
            "eval_func": "classification_single_func",
            "train_only": False,
            "eval_only": False,
            "class_emb_flag": False,
            "random_flag": True,
    },
    "arxiv_zs": {
            "dataset_name": "arxiv",
            "task_level": "node",
            "construct": "ConstructNCFSZS",
            "args": {"walk_length": None, "n_way": 5, "min_n": 3, "val_n": [5, 3], "k_shot": 0, "min_k": 0, "val_k": [0], "q_query": 3, "fs_task_num": 5, "class_split_ratio": None, },
            "mode": {"train": 0, "valid":1, "test":2},
            "eval_set_constructs": [
                {
                    "stage": "valid",
                },
                {
                    "stage": "test",
                },
            ],
            "eval_metric": "acc",
            "eval_func": "classification_single_func",
            "train_only": False,
            "eval_only": False,
            "class_emb_flag": True,
            "random_flag": True,
    },
    "cora_fs": {
            "dataset_name": "cora",
            "task_level": "node",
            "construct": "ConstructNCFSZS",
            "args": {"walk_length": None, "n_way": 5, "min_n": 2, "val_n": [5, 2], "k_shot": 5, "min_k": 1, "val_k": [1, 3, 5], "q_query": 1, "fs_task_num": 1, "class_split_ratio": [0, 0, 7], },
            "mode": {"valid":2, "test":2},
            "eval_set_constructs": [
                {
                    "stage": "valid",
                },
                {
                    "stage": "test",
                },
            ],
            "eval_metric": "acc",
            "eval_func": "classification_single_func",
            "train_only": False,
            "eval_only": True,
            "class_emb_flag": False,
    },
    "cora_zs": {
            "dataset_name": "cora",
            "task_level": "node",
            "construct": "ConstructNCFSZS",
            "args": {"walk_length": None, "n_way": 5, "min_n": 2, "val_n": [5, 2], "k_shot": 5, "min_k": 1, "val_k": [1, 3, 5], "q_query": 1, "fs_task_num": 1, "class_split_ratio": [0, 0, 7], },
            "mode": {"valid":2, "test":2},
            "eval_set_constructs": [
                {
                    "stage": "valid",
                },
                {
                    "stage": "test",
                },
            ],
            "eval_metric": "acc",
            "eval_func": "classification_single_func",
            "train_only": False,
            "eval_only": True,
            "class_emb_flag": True,
    },
    "fb_fs": {
            "dataset_name": "FB15K237",
            "task_level": "link",
            "construct": "ConstructLPFSZS",
            "args": {"walk_length": None, "n_way": 10, "min_n": 5, "val_n": [10, 5], "k_shot": 5, "min_k": 1, "val_k": [1, 3, 5], "q_query": 1, "fs_task_num": 1, "class_split_ratio": None, },
            "mode": {"train": 0, "valid":1, "test":2},
            "eval_set_constructs": [
                {
                    "stage": "valid",
                },
                {
                    "stage": "test",
                },
            ],
            "eval_metric": "acc",
            "eval_func": "classification_single_func",
            "train_only": False,
            "eval_only": False,
            "class_emb_flag": False,
            "random_flag": True,
    },
    "fb_zs": {
            "dataset_name": "FB15K237",
            "task_level": "link",
            "construct": "ConstructLPFSZS",
            "args": {"walk_length": None, "n_way": 10, "min_n": 5, "val_n": [10, 5], "k_shot": 0, "min_k": 0, "val_k": [0], "q_query": 1, "fs_task_num": 1, "class_split_ratio": None, },
            "mode": {"train": 0, "valid":1, "test":2},
            "eval_set_constructs": [
                {
                    "stage": "valid",
                },
                {
                    "stage": "test",
                },
            ],
            "eval_metric": "acc",
            "eval_func": "classification_single_func",
            "train_only": False,
            "eval_only": False,
            "class_emb_flag": True,
            "random_flag": True,
    },
    "wn_fs": {
            "dataset_name": "WN18RR",
            "task_level": "link",
            "construct": "ConstructLPFSZS",
            "args": {"walk_length": None, "n_way": 10, "min_n": 5, "val_n": [10, 5], "k_shot": 5, "min_k": 1, "val_k": [1, 3, 5], "q_query": 1, "fs_task_num": 1, "class_split_ratio": [0, 0, 11], },
            "mode": {"valid":2, "test":2},
            "eval_set_constructs": [
                {
                    "stage": "valid",
                },
                {
                    "stage": "test",
                },
            ],
            "eval_metric": "acc",
            "eval_func": "classification_single_func",
            "train_only": False,
            "eval_only": True,
            "class_emb_flag": False,
    },
    "wn_zs": {
            "dataset_name": "WN18RR",
            "task_level": "link",
            "construct": "ConstructLPFSZS",
            "args": {"walk_length": None, "n_way": 10, "min_n": 5, "val_n": [10, 5], "k_shot": 0, "min_k": 0, "val_k": [0], "q_query": 1, "fs_task_num": 1, "class_split_ratio": [0, 0, 11], },
            "mode": {"valid":2, "test":2},
            "eval_set_constructs": [
                {
                    "stage": "valid",
                },
                {
                    "stage": "test",
                },
            ],
            "eval_metric": "acc",
            "eval_func": "classification_single_func",
            "train_only": False,
            "eval_only": True,
            "class_emb_flag": True,
    },
    "chemblpre_fs": {
        "dataset_name": "chemblpre",
        "dataset_splitter": "MolSplitter",
        "preprocess": None,
        "construct": "ConstructMolCls",
        "args": {"walk_length": None},
        "process_label_func": "process_multi_label",
        "eval_set_constructs": [
            {
                "stage": "valid",
                "split_name": "valid",
                "process_label_func": "none_process_label",
            },
            {
                "stage": "test",
                "split_name": "test",
                "process_label_func": "none_process_label",
            },
            {
                "stage": "test",
                "split_name": "train",
                "process_label_func": "none_process_label",
            },
        ],
        "eval_metric": "apr",
        "eval_func": "binary_apr_func",
        "num_classes": 1296,
        "train_only": True,
        "eval_only": False,
    },
}


class TaskConstructor:
    def __init__(self, tasks, encoder, task_config_lookup, batch_size=256, sample_size=-1):
        self.tasks = tasks
        self.task_config_lookup = task_config_lookup
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.dataset = {}
        self.train_set = []
        self.valid_dm_set = []
        self.test_dm_set = []

        for task in self.tasks:
            config = self.task_config_lookup[task]
            data = config["dataset_name"]
            if data not in self.dataset and data in name2dataset:
                self.dataset[data] = name2dataset[data](
                    data, sentence_encoder=encoder
                )

            split = globals()[config["dataset_splitter"]](self.dataset[data])
            if config["preprocess"] is not None:
                global_data = globals()[config["preprocess"]](
                    self.dataset[data], split
                )
            else:
                global_data = None

            train_data = globals()[config["construct"]](
                task,
                self.dataset[data],
                split,
                "train",
                globals()[config["process_label_func"]],
                global_data=global_data,
                **config["args"],
            )
            self.train_set.append(train_data)

            for eval_construct_config in config["eval_set_constructs"]:
                if "process_label_func" in eval_construct_config:
                    trim_class_func = globals()[
                        eval_construct_config["process_label_func"]
                    ]
                else:
                    trim_class_func = globals()[config["process_label_func"]]

                if "args" in eval_construct_config:
                    eval_args = eval_construct_config["args"]
                else:
                    eval_args = config["args"]

                if "construct" in eval_construct_config:
                    construct = globals()[eval_construct_config["construct"]]
                else:
                    construct = globals()[config["construct"]]
                eval_data = construct(
                    task,
                    self.dataset[data],
                    split,
                    eval_construct_config["split_name"],
                    trim_class_func,
                    global_data=global_data,
                    **eval_args,
                )

                dm_data = make_data(
                    data,
                    eval_data,
                    eval_construct_config["split_name"],
                    config["eval_metric"],
                    globals()[config["eval_func"]],
                    config["num_classes"],
                    batch_size=self.batch_size,
                    sample_size=self.sample_size,
                )

                if eval_construct_config["stage"] == "valid":
                    self.valid_dm_set.append(dm_data)
                else:
                    self.test_dm_set.append(dm_data)

    def make_train_data(self, multiple, min_ratio):
        train_data = MultiDataset(
            self.train_set,
            dataset_multiple=multiple,
            patience=3,
            window_size=5,
            min_ratio=min_ratio,
        )
        return train_data

    def make_full_dm_list(self, multiple, min_ratio, train_data=None):
        text_dataset = {
            "train": DataWithMeta(
                self.make_train_data(multiple, min_ratio)
                if not train_data
                else train_data,
                self.batch_size,
                sample_size=self.sample_size,
            ),
            "val": self.valid_dm_set,
            "test": self.test_dm_set,
        }
        return text_dataset


class LowResourceTaskConstructor(TaskConstructor):
    def __init__(self, tasks, encoder, batch_size=3, sample_size=-1):
        self.tasks = tasks
        self.encoder = encoder
        self.batch_size = batch_size
        self.sample_size = sample_size
        with open('data/low_resource_split.json', 'r') as f:
            self.lr_class_split = json.load(f)

        self.dataset = {}
        self.datamanager = {}
        self.edges = {}
        self.train_set = []
        self.valid_dm_set = []
        self.test_dm_set = []

        for task in self.tasks:
            # get task config
            config = task_config_lookup[task]
            data = config["dataset_name"]
            if data not in self.dataset and data in name2dataset:
                self.dataset[data] = name2dataset[data](
                    data, sentence_encoder=encoder
                )

            # save train data to self.train_set
            if config["task_level"] == "node" or config["task_level"] == "link":
                g = self.get_lr_traindata(data, config)
            else:
                split, global_data = self.get_traindata(data, config, task)


            if not config["train_only"]:
                for eval_construct_config in config["eval_set_constructs"]:
                    if "args" in eval_construct_config:
                        eval_args = eval_construct_config["args"]
                    else:
                        eval_args = config["args"]

                    if "construct" in eval_construct_config:
                        construct = globals()[eval_construct_config["construct"]]
                    else:
                        construct = globals()[config["construct"]]

                    if config["task_level"] == "node" or config["task_level"] == "link":
                        eval_data = self.get_lr_evaldata(construct, data, g, config, eval_construct_config, eval_args)
                    else:
                        eval_data = self.get_evaldata(construct, data, task, split, global_data, config, eval_construct_config, eval_args)

                    if eval_construct_config["stage"] == "valid":
                        self.valid_dm_set += eval_data
                    else:
                        self.test_dm_set += eval_data

    def get_node_graph(self, dataset):
        g = dataset.data
        g.x = g.x_text_feat
        return g

    def get_link_graph(self, dataset, data, class_split_lst=None):
        g = dataset.data
        g.x = g.x_text_feat

        if data not in self.edges and data in name2dataset:
            self.edges[data] = {}
            converted_triplet = dataset.get_idx_split()
            edges = torch.cat(
                [
                    torch.tensor(converted_triplet["train"][0]).T,
                    torch.tensor(converted_triplet["valid"][0]).T,
                    torch.tensor(converted_triplet["test"][0]).T,
                ],
                dim=-1,
            )
            self.edges[data]["edges"] = edges
            self.edges[data]["fs_edges"] = [[], [], edges]
            edge_labels = torch.cat(
                [
                    torch.tensor(converted_triplet["train"][1]),
                    torch.tensor(converted_triplet["valid"][1]),
                    torch.tensor(converted_triplet["test"][1]),
                ]
            )
            self.edges[data]["edge_labels"] = edge_labels
            self.edges[data]["fs_edge_types"] = [[], [], edge_labels]
            if class_split_lst is not None:
                fs_edges = []
                fs_edge_types = []
                for classes in class_split_lst:
                    fs_mask = torch.tensor(
                        [item in classes for item in edge_labels]
                    )
                    fs_edges.append(edges[:, fs_mask])
                    fs_edge_types.append(edge_labels[fs_mask])
                self.edges[data]["fs_edges"] = fs_edges
                self.edges[data]["fs_edge_types"] = fs_edge_types

        g.edge_index = self.edges[data]["edges"]
        g.y = self.edges[data]["edge_labels"]
        return g

    def get_lr_traindata(self, data, config):
        g = self.get_node_graph(self.dataset[data]) if config["task_level"] == "node" else self.get_link_graph(
            self.dataset[data], data, self.lr_class_split.get(data)) if config["task_level"] == "link" else None

        args = config["args"]
        if data in self.edges:
            args["edges"] = self.edges[data]
        if data not in self.datamanager and data in name2dataset:
            self.datamanager[data] = FewShotDataManager(g, args["n_way"], args["k_shot"], args["q_query"], class_split_ratio=args["class_split_ratio"], class_split_lst=self.lr_class_split.get(data))

        if not config["eval_only"]:
            train_data = globals()[config["construct"]](
                g,
                self.datamanager[data],
                n=args["min_n"],
                k=args["min_k"],
                split_name=config["mode"]["train"],
                random_flag=config["random_flag"],
                train_flag=True,
                class_emb_flag=config["class_emb_flag"],
                **args,
            )
            self.train_set.append(train_data)
        return g

    def get_lr_evaldata(self, construct, data, g, config, eval_construct_config, eval_args):
        eval_data = [
            construct(
                g,
                self.datamanager[data],
                n=n,
                k=k,
                split_name=config["mode"][eval_construct_config["stage"]],
                state_name=f'{eval_construct_config["stage"]}_fs{n}{k}_{data}',
                eval_metric=config["eval_metric"],
                eval_func=globals()[config["eval_func"]],
                class_emb_flag=config["class_emb_flag"],
                **eval_args,
            )
            for n in eval_args["val_n"]
            for k in eval_args["val_k"]
        ]
        return eval_data

    def get_traindata(self, data, config, task):
        split = globals()[config["dataset_splitter"]](self.dataset[data])
        if config["preprocess"] is not None:
            global_data = globals()[config["preprocess"]](
                self.dataset[data], split
            )
        else:
            global_data = None

        train_data = globals()[config["construct"]](
            task,
            self.dataset[data],
            split,
            "train",
            globals()[config["process_label_func"]],
            global_data=global_data,
            **config["args"],
        )
        self.train_set.append(train_data)
        return split, global_data

    def get_evaldata(self, construct, data, task, split, global_data, config, eval_construct_config, eval_args):
        if "process_label_func" in eval_construct_config:
            trim_class_func = globals()[
                eval_construct_config["process_label_func"]
            ]
        else:
            trim_class_func = globals()[config["process_label_func"]]

        eval_data = construct(
            task,
            self.dataset[data],
            split,
            eval_construct_config["split_name"],
            trim_class_func,
            global_data=global_data,
            **eval_args,
        )

        dm_data = make_data(
            data,
            eval_data,
            eval_construct_config["split_name"],
            config["eval_metric"],
            globals()[config["eval_func"]],
            config["num_classes"],
            batch_size=self.batch_size,
            sample_size=self.sample_size,
        )

        return dm_data

