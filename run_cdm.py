import argparse
import os
import torch
import torch_geometric as pyg
from pytorch_lightning.loggers import WandbLogger
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
from gp.utils.utils import (
    load_yaml,
    combine_dict,
    merge_mod,
    setup_exp,
    set_random_seed,
    k_fold2_split,
    k_fold_ind,
)
from gp.lightning.metric import (
    binary_auc_func,
    flat_binary_func,
    classification_func,
    EvalKit,
)
from gp.lightning.data_template import DataModule, DataWithMeta
from gp.lightning.training import lightning_fit, lightning_test
from gp.lightning.module_template import ExpConfig
from types import SimpleNamespace
from lightning_model import GraphPredLightning
from models.model import BinGraphModel, BinGraphAttModel, AdaPoolClassModel
from gp.nn.models.pyg import PyGGIN, PyGRGCN, PyGGINE
from models.model import PyGRGCNEdge

from torchmetrics import AUROC, Accuracy
from utils import (
    SentenceEncoder,
    binary_apr_func,
    MultiApr,
    flat_label_func,
    label_apr_func,
    binary_single_auc_func,
    binary_auc_multi_func,
    MultiAuc,
)
from data.arxiv.gen_data import ArxivOFADataset
from data.Cora.gen_data import CoraOFADataset
from data.Pubmed.gen_data import PubmedOFADataset
from data.WN18RR.gen_data import WN18RROFADataset
from data.FB15K237.gen_data import FB15K237OFADataset
from data.wikics.gen_data import WikiCSOFADataset
from data.chemblpre.gen_data import CHEMBLPREOFADataset
from data.chempcba.gen_data import CHEMPCBAOFADataset
from data.chemhiv.gen_data import CHEMHIVOFADataset

from gp.lightning.metric import flat_binary_func_fs
from gp.utils.utils import SmartTimer
from scipy.sparse import csr_array


def data_construct(
    data_names,
    encoder,
    batch_size=None,
    sample_size=None,
    walk_length=None,
    fs_setting=None,
):
    constructed_data = []
    for data_idx, name in enumerate(data_names):
        print("loading: ", name)
        if name == "arxiv":
            dataset = ArxivOFADataset("arxiv", sentence_encoder=encoder)
            text_g = dataset.data
            text_g.x = text_g.x_text_feat
            text_g.prompt_edge_feat = dataset.prompt_edge_feat
            kfold = k_fold_ind(text_g.y, 10)
            text_split = k_fold2_split(kfold, len(text_g.y))[0]

            def trim_class(label, num_class):
                binary_rep = torch.zeros((1, num_class))
                binary_rep[0, label] = 1
                return label.view(1, -1), binary_rep

            def make_data(split_name, state_name):
                return DataWithMeta(
                    SubgraphHierDataset(
                        text_g,
                        text_g.label_text_feat,
                        text_split[split_name],
                        prompt_feat=text_g.prompt_text_feat,
                        to_undirected=True,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="acc",
                    state_name=state_name,
                    classes=40,
                    meta_data={"eval_func": classification_func},
                )

            split_data = {
                "train": [
                    SubgraphHierDataset(
                        text_g,
                        text_g.label_text_feat,
                        text_split[0],
                        prompt_feat=text_g.prompt_text_feat,
                        to_undirected=True,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    )
                ],
                "test": [
                    make_data(2, "test_arxiv"),
                    make_data(0, "test_train_arxiv"),
                ],
                "val": [make_data(1, "valid_arxiv")],
            }
            constructed_data.append(split_data)

        if name == "arxivnohier":
            dataset = ArxivOFADataset("arxiv", sentence_encoder=encoder)
            text_g = dataset.data
            text_g.x = text_g.x_text_feat
            text_g.prompt_edge_feat = dataset.prompt_edge_feat
            kfold = k_fold_ind(text_g.y, 10)
            text_split = k_fold2_split(kfold, len(text_g.y))[0]

            def trim_class(label, num_class):
                binary_rep = torch.zeros((1, num_class))
                binary_rep[0, label] = 1
                return label.view(1, -1), binary_rep

            def make_data(split_name, state_name):
                return DataWithMeta(
                    SubgraphDataset(
                        text_g,
                        text_g.label_text_feat,
                        text_split[split_name],
                        to_undirected=True,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="acc",
                    state_name=state_name,
                    classes=40,
                    meta_data={"eval_func": classification_func},
                )

            split_data = {
                "train": [
                    SubgraphDataset(
                        text_g,
                        text_g.label_text_feat,
                        text_split[0],
                        to_undirected=True,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    )
                ],
                "test": [
                    make_data(2, "test_arxiv"),
                    make_data(0, "test_train_arxiv"),
                ],
                "val": [make_data(1, "valid_arxiv")],
            }
            constructed_data.append(split_data)
        if name == "arxivnoprompt":
            dataset = ArxivOFADataset("arxiv", sentence_encoder=encoder)
            text_g = dataset.data
            text_g.x = text_g.x_text_feat
            text_g.prompt_edge_feat = dataset.prompt_edge_feat
            kfold = k_fold_ind(text_g.y, 10)
            text_split = k_fold2_split(kfold, len(text_g.y))[0]

            def trim_class(label, num_class):
                binary_rep = torch.zeros((1, num_class))
                binary_rep[0, label] = 1
                return label.view(1, -1), binary_rep

            def make_data(split_name, state_name):
                return DataWithMeta(
                    SubgraphNopromptDataset(
                        text_g,
                        text_g.label_text_feat,
                        text_split[split_name],
                        to_undirected=True,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="acc",
                    state_name=state_name,
                    classes=40,
                    meta_data={"eval_func": classification_func},
                )

            split_data = {
                "train": [
                    SubgraphNopromptDataset(
                        text_g,
                        text_g.label_text_feat,
                        text_split[0],
                        to_undirected=True,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    )
                ],
                "test": [
                    make_data(2, "test_arxiv"),
                    make_data(0, "test_train_arxiv"),
                ],
                "val": [make_data(1, "valid_arxiv")],
            }
            constructed_data.append(split_data)

        if name in ["chempcba", "chemblpre"]:
            if name == "chempcba":
                dataset = CHEMPCBAOFADataset(
                    "chempcba", sentence_encoder=encoder
                )
            elif name == "chemblpre":
                dataset = CHEMBLPREOFADataset(
                    "chemblpre", sentence_encoder=encoder
                )
            # print(dataset[350342])
            # return 0
            split = dataset.get_idx_split()

            def trim_class(embs, classes):
                valid_idx = classes == classes
                # valid_idx = torch.zeros_like(classes, dtype=torch.bool)
                return (
                    torch.tensor([[0]]),
                    embs[valid_idx.view(-1)].detach().clone(),
                    classes[:, valid_idx.view(-1)].detach().clone(),
                )

            def make_data(split_name, state_name):
                return DataWithMeta(
                    GraphListHierDataset(
                        dataset,
                        dataset.label_text_feat,
                        dataset.prompt_edge_feat,
                        dataset.prompt_text_feat,
                        split[split_name],
                        single_prompt_edge=True,
                        # trim_class_func=trim_class,
                        walk_length=walk_length,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="apr",
                    state_name=state_name,
                    classes=len(dataset.label_text_feat),
                    meta_data={"eval_func": binary_apr_func},
                )

            split_data = {
                "train": [
                    GraphListHierDataset(
                        dataset,
                        dataset.label_text_feat,
                        dataset.prompt_edge_feat,
                        dataset.prompt_text_feat,
                        split["train"],
                        trim_class_func=trim_class,
                        single_prompt_edge=True,
                        walk_length=walk_length,
                    )
                ],
                "test": [
                    make_data("test", "test_" + name),
                    make_data("train", "test_train_" + name),
                ],
                "val": [make_data("valid", "valid_" + name)],
            }
            constructed_data.append(split_data)
        if name in ["chemhiv"]:
            if name == "chemhiv":
                dataset = CHEMHIVOFADataset(
                    "chemhiv", sentence_encoder=encoder
                )
            # print(dataset[350342])
            # return 0
            split = dataset.get_idx_split()

            # def trim_class(embs, classes):
            #     valid_idx = classes == classes
            #     # valid_idx = torch.zeros_like(classes, dtype=torch.bool)
            #     return (
            #         torch.tensor([[0]]),
            #         embs[valid_idx.view(-1)].detach().clone(),
            #         classes[:, valid_idx.view(-1)].detach().clone(),
            #     )

            def trim_class(embs, label):
                label = label.to(torch.long)
                one_hot_label = torch.nn.functional.one_hot(
                    label, num_classes=2
                )
                return label, embs, one_hot_label

            # print(sub_dataset.label_text_feat.size())

            def make_data(split_name, state_name):
                return DataWithMeta(
                    GraphListHierDataset(
                        dataset,
                        dataset.label_text_feat,
                        dataset.prompt_edge_feat,
                        dataset.prompt_text_feat,
                        split[split_name],
                        single_prompt_edge=True,
                        # trim_class_func=trim_class,
                        walk_length=walk_length,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="auc",
                    state_name=state_name,
                    classes=len(dataset.label_text_feat),
                    meta_data={"eval_func": binary_auc_func},
                )

            split_data = {
                "train": [
                    GraphListHierDataset(
                        dataset,
                        dataset.label_text_feat,
                        dataset.prompt_edge_feat,
                        dataset.prompt_text_feat,
                        split["train"],
                        trim_class_func=trim_class,
                        single_prompt_edge=True,
                        walk_length=walk_length,
                    )
                ],
                "test": [
                    make_data("test", "test_" + name),
                    make_data("train", "test_train_" + name),
                ],
                "val": [make_data("valid", "valid_" + name)],
            }
            constructed_data.append(split_data)

        if name in ["chemhivnohier"]:
            if name == "chemhivnohier":
                dataset = CHEMHIVOFADataset(
                    "chemhiv", sentence_encoder=encoder
                )
            # print(dataset[350342])
            # return 0
            split = dataset.get_idx_split()

            # def trim_class(embs, classes):
            #     valid_idx = classes == classes
            #     # valid_idx = torch.zeros_like(classes, dtype=torch.bool)
            #     return (
            #         torch.tensor([[0]]),
            #         embs[valid_idx.view(-1)].detach().clone(),
            #         classes[:, valid_idx.view(-1)].detach().clone(),
            #     )

            def trim_class(embs, label):
                label = label.to(torch.long)
                one_hot_label = torch.nn.functional.one_hot(
                    label, num_classes=2
                )
                return label, embs, one_hot_label

            # print(sub_dataset.label_text_feat.size())

            def make_data(split_name, state_name):
                return DataWithMeta(
                    GraphListDataset(
                        dataset,
                        dataset.label_text_feat,
                        dataset.prompt_edge_feat,
                        split[split_name],
                        single_prompt_edge=True,
                        # trim_class_func=trim_class,
                        walk_length=walk_length,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="auc",
                    state_name=state_name,
                    classes=len(dataset.label_text_feat),
                    meta_data={"eval_func": binary_auc_func},
                )

            split_data = {
                "train": [
                    GraphListDataset(
                        dataset,
                        dataset.label_text_feat,
                        dataset.prompt_edge_feat,
                        split["train"],
                        trim_class_func=trim_class,
                        single_prompt_edge=True,
                        walk_length=walk_length,
                    )
                ],
                "test": [
                    make_data("test", "test_" + name),
                    make_data("train", "test_train_" + name),
                ],
                "val": [make_data("valid", "valid_" + name)],
            }
            constructed_data.append(split_data)

        if name in ["chemhivnoprompt"]:
            if name == "chemhivnoprompt":
                dataset = CHEMHIVOFADataset(
                    "chemhiv", sentence_encoder=encoder
                )
            # print(dataset[350342])
            # return 0
            split = dataset.get_idx_split()

            # def trim_class(embs, classes):
            #     valid_idx = classes == classes
            #     # valid_idx = torch.zeros_like(classes, dtype=torch.bool)
            #     return (
            #         torch.tensor([[0]]),
            #         embs[valid_idx.view(-1)].detach().clone(),
            #         classes[:, valid_idx.view(-1)].detach().clone(),
            #     )

            def trim_class(embs, label):
                label = label.to(torch.long)
                one_hot_label = torch.nn.functional.one_hot(
                    label, num_classes=2
                )
                return label, embs, one_hot_label

            # print(sub_dataset.label_text_feat.size())

            def make_data(split_name, state_name):
                return DataWithMeta(
                    GraphListNopromptDataset(
                        dataset,
                        dataset.label_text_feat,
                        dataset.prompt_edge_feat,
                        split[split_name],
                        single_prompt_edge=True,
                        # trim_class_func=trim_class,
                        walk_length=walk_length,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="auc",
                    state_name=state_name,
                    classes=len(dataset.label_text_feat),
                    meta_data={"eval_func": binary_auc_func},
                )

            split_data = {
                "train": [
                    GraphListNopromptDataset(
                        dataset,
                        dataset.label_text_feat,
                        dataset.prompt_edge_feat,
                        split["train"],
                        trim_class_func=trim_class,
                        single_prompt_edge=True,
                        walk_length=walk_length,
                    )
                ],
                "test": [
                    make_data("test", "test_" + name),
                    make_data("train", "test_train_" + name),
                ],
                "val": [make_data("valid", "valid_" + name)],
            }
            constructed_data.append(split_data)

        if name in ["coralink", "pubmedlink"]:
            if name == "coralink":
                dataset = CoraOFADataset("cora", sentence_encoder=encoder)
            else:
                dataset = PubmedOFADataset("pubmed", sentence_encoder=encoder)
            text_g = dataset.data
            text_g.x = text_g.x_text_feat
            text_g.prompt_edge_feat = dataset.prompt_edge_feat
            edges = text_g.edge_index
            edge_perm = torch.randperm(len(edges[0]))
            train_offset = int(len(edge_perm) * 0.85)
            val_offset = int(len(edge_perm) * 0.9)
            edge_indices = {
                "train": edge_perm[:train_offset],
                "val": edge_perm[train_offset:val_offset],
                "test": edge_perm[val_offset:],
            }
            graph_dict = text_g.to_dict()
            graph_dict["edge_index"] = edges[:, edge_indices["train"]]
            train_graph = pyg.data.Data(**graph_dict)

            def trim_class(label, num_class):
                binary_rep = torch.zeros((1, num_class))
                binary_rep[0, label] = 1
                return torch.tensor([label]).view(1, -1), binary_rep

            def make_data(split_name, state_name, remove_edge=False):
                return DataWithMeta(
                    SubgraphLinkHierDataset(
                        train_graph,
                        train_graph.edge_label_feat,
                        edges.T[edge_indices[split_name]].numpy(),
                        prompt_feat=train_graph.prompt_node_edge_feat,
                        to_undirected=True,
                        hop=3,
                        remove_edge=remove_edge,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="auc",
                    state_name=state_name,
                    classes=2,
                    meta_data={"eval_func": binary_auc_func},
                )

            split_data = {
                "train": [
                    SubgraphLinkHierDataset(
                        train_graph,
                        train_graph.edge_label_feat,
                        edges.T[edge_indices["train"]].numpy(),
                        prompt_feat=train_graph.prompt_node_edge_feat,
                        to_undirected=True,
                        hop=3,
                        remove_edge=True,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    )
                ],
                "test": [
                    make_data("test", "test_" + name, remove_edge=False),
                    make_data("train", "test_train_" + name, remove_edge=True),
                ],
                "val": [make_data("val", "valid_" + name, remove_edge=False)],
            }
            constructed_data.append(split_data)

        if name in ["coranode", "pubmednode"]:
            if name == "coranode":
                dataset = CoraOFADataset("cora", sentence_encoder=encoder)
            else:
                dataset = PubmedOFADataset("pubmed", sentence_encoder=encoder)
            text_g = dataset.data
            text_g.x = text_g.x_text_feat
            text_g.prompt_edge_feat = dataset.prompt_edge_feat
            split = {
                "train": text_g.train_masks[0].nonzero(as_tuple=True)[0],
                "val": text_g.val_masks[0].nonzero(as_tuple=True)[0],
                "test": text_g.test_masks[0].nonzero(as_tuple=True)[0],
            }

            def trim_class(label, num_class):
                binary_rep = torch.zeros((1, num_class))
                binary_rep[0, label] = 1
                return torch.tensor([label]).view(1, -1), binary_rep

            def make_data(split_name, state_name):
                return DataWithMeta(
                    SubgraphHierDataset(
                        text_g,
                        text_g.label_text_feat,
                        split[split_name],
                        prompt_feat=text_g.prompt_node_feat,
                        to_undirected=True,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="acc",
                    state_name=state_name,
                    classes=len(text_g.label_text_feat),
                    meta_data={"eval_func": classification_func},
                )

            split_data = {
                "train": [
                    SubgraphHierDataset(
                        text_g,
                        text_g.label_text_feat,
                        split["train"],
                        prompt_feat=text_g.prompt_node_feat,
                        to_undirected=True,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    )
                ],
                "test": [
                    make_data("test", "test_" + name),
                    make_data("train", "test_train_" + name),
                ],
                "val": [make_data("val", "valid_" + name)],
            }
            constructed_data.append(split_data)

        if name in ["WN18RR", "FB15K237"]:
            if name == "WN18RR":
                dataset = WN18RROFADataset("WN18RR", sentence_encoder=encoder)
            else:
                dataset = FB15K237OFADataset(
                    "FB15K237", sentence_encoder=encoder
                )
            text_g = dataset.data
            text_g.x = text_g.x_text_feat
            text_g.prompt_edge_feat = dataset.prompt_edge_feat
            edges = text_g.edge_index
            converted_triplet = dataset.get_idx_split()

            def trim_class(label, num_class):
                binary_rep = torch.zeros((1, num_class))
                binary_rep[0, label] = 1
                return torch.tensor([label]).view(1, -1), binary_rep

            def make_data(split_name, state_name, remove_edge=False):
                return DataWithMeta(
                    SubgraphKGHierDataset(
                        text_g,
                        text_g.edge_label_feat,
                        converted_triplet[split_name],
                        prompt_feat=text_g.prompt_text_feat,
                        to_undirected=True,
                        hop=2,
                        remove_edge=remove_edge,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="acc",
                    state_name=state_name,
                    classes=len(text_g.edge_label_feat),
                    meta_data={"eval_func": classification_func},
                )

            split_data = {
                "train": [
                    SubgraphKGHierDataset(
                        text_g,
                        text_g.edge_label_feat,
                        converted_triplet["train"],
                        prompt_feat=text_g.prompt_text_feat,
                        to_undirected=True,
                        hop=2,
                        remove_edge=True,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    )
                ],
                "test": [
                    make_data("test", "test_" + name, remove_edge=False),
                    make_data("train", "test_train_" + name, remove_edge=True),
                ],
                "val": [
                    make_data("valid", "valid_" + name, remove_edge=False)
                ],
            }
            constructed_data.append(split_data)

        if name == "wikics":
            dataset = WikiCSOFADataset("wikics", sentence_encoder=encoder)
            text_g = dataset.data
            text_g.x = text_g.x_text_feat
            text_g.prompt_edge_feat = dataset.prompt_edge_feat
            # wikics provides 20 split combinations for train and val
            wiki_split_idx = 0
            text_split = [
                torch.where(text_g.train_mask[:, wiki_split_idx])[0].numpy(),
                torch.where(text_g.val_mask[:, wiki_split_idx])[0].numpy(),
                torch.where(text_g.test_mask)[0].numpy(),
            ]

            def trim_class(label, num_class):
                binary_rep = torch.zeros((1, num_class))
                binary_rep[0, label] = 1
                return label.view(1, -1), binary_rep

            def make_data(split_name, state_name):
                return DataWithMeta(
                    SubgraphHierDataset(
                        text_g,
                        text_g.label_text_feat,
                        text_split[split_name],
                        prompt_feat=text_g.prompt_text_feat,
                        to_undirected=False,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="acc",
                    state_name=state_name,
                    classes=10,
                    meta_data={"eval_func": classification_func},
                )

            split_data = {
                "train": [
                    SubgraphHierDataset(
                        text_g,
                        text_g.label_text_feat,
                        text_split[0],
                        prompt_feat=text_g.prompt_text_feat,
                        to_undirected=False,
                        trim_class_func=trim_class,
                        walk_length=walk_length,
                    )
                ],
                "test": [
                    make_data(2, "test_wikics"),
                    make_data(0, "test_train_wikics"),
                ],
                "val": [make_data(1, "valid_wikics")],
            }
            constructed_data.append(split_data)

        if name == "molzero":
            chembl_dataset = CHEMBLPREOFADataset(
                "chemblpre", sentence_encoder=encoder
            )
            hiv_dataset = CHEMHIVOFADataset(
                "chemhiv", sentence_encoder=encoder
            )
            pcba_dataset = CHEMPCBAOFADataset(
                "chempcba", sentence_encoder=encoder
            )
            hiv_split = hiv_dataset.get_idx_split()
            chembl_split = chembl_dataset.get_idx_split()
            pcba_split = pcba_dataset.get_idx_split()

            def trim_class(embs, classes):
                valid_idx = classes == classes
                # valid_idx = torch.zeros_like(classes, dtype=torch.bool)
                return (
                    torch.tensor([[0]]),
                    embs[valid_idx.view(-1)].detach().clone(),
                    classes[:, valid_idx.view(-1)].detach().clone(),
                )

            def hiv_trim_class(embs, label):
                # one_hot_label = torch.nn.functional.one_hot(
                #     label.to(torch.long), num_classes=2
                # )
                return label, embs[0:1], label

            def make_data(split_name, state_name):
                return DataWithMeta(
                    GraphListHierDataset(
                        hiv_dataset,
                        hiv_dataset.label_text_feat,
                        hiv_dataset.prompt_edge_feat,
                        hiv_dataset.prompt_text_feat,
                        hiv_split[split_name],
                        trim_class_func=hiv_trim_class,
                        single_prompt_edge=True,
                        walk_length=walk_length,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="auc",
                    state_name=state_name,
                    classes=1,
                    meta_data={"eval_func": binary_single_auc_func},
                )

            def make_pcba_data(split_name, state_name):
                return DataWithMeta(
                    GraphListHierDataset(
                        pcba_dataset,
                        pcba_dataset.label_text_feat,
                        pcba_dataset.prompt_edge_feat,
                        pcba_dataset.prompt_text_feat,
                        pcba_split[split_name],
                        # trim_class_func=trim_class,
                        single_prompt_edge=True,
                        walk_length=walk_length,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="aucmulti",
                    state_name=state_name,
                    classes=128,
                    meta_data={"eval_func": binary_auc_multi_func},
                )

            split_data = {
                "train": [
                    GraphListHierDataset(
                        chembl_dataset,
                        chembl_dataset.label_text_feat,
                        chembl_dataset.prompt_edge_feat,
                        chembl_dataset.prompt_text_feat,
                        chembl_split["train"],
                        trim_class_func=trim_class,
                        single_prompt_edge=True,
                        walk_length=walk_length,
                    )
                ],
                "test": [
                    make_data("test", "test_molhiv"),
                    # make_data("train", "test_train_molhiv"),
                    make_pcba_data("test", "test_pcba"),
                ],
                "val": [
                    make_data("valid", "valid_molhiv"),
                    make_pcba_data("valid", "valid_pcba"),
                ],
            }
            constructed_data.append(split_data)
        if name == "molfew":
            chembl_dataset = CHEMBLPREOFADataset(
                "chemblpre", sentence_encoder=encoder
            )
            hiv_dataset = CHEMHIVOFADataset(
                "chemhiv", sentence_encoder=encoder
            )
            pcba_dataset = CHEMPCBAOFADataset(
                "chempcba", sentence_encoder=encoder
            )
            pcba_split = pcba_dataset.get_idx_split()
            hiv_split = hiv_dataset.get_idx_split()
            chembl_split = chembl_dataset.get_idx_split()
            chembl_classes = chembl_dataset.y.view(len(chembl_dataset), -1)[
                chembl_split["train"]
            ]

            pcba_classes = [49, 60, 47, 94, 93]
            shots = [1, 3, 5, 10]

            def trim_class(embs, classes):
                valid_idx = classes == classes
                # valid_idx = torch.zeros_like(classes, dtype=torch.bool)
                return (
                    torch.tensor([[0]]),
                    embs[valid_idx.view(-1)].detach().clone(),
                    classes[:, valid_idx.view(-1)].detach().clone(),
                )

            def hiv_trim_class(embs, label):
                one_hot_label = torch.nn.functional.one_hot(
                    label.to(torch.long), num_classes=2
                )
                return label, embs, one_hot_label

            def make_data(split_name, state_name, shot=1):
                return DataWithMeta(
                    GraphListHierFSDataset(
                        hiv_dataset,
                        hiv_dataset.label_text_feat,
                        hiv_dataset.prompt_edge_feat,
                        hiv_dataset.prompt_text_feat,
                        hiv_split[split_name],
                        trim_class_func=hiv_trim_class,
                        single_prompt_edge=True,
                        walk_length=walk_length,
                        class_ind=hiv_dataset.y.view(len(hiv_dataset), -1)[
                            hiv_split[split_name], 0:1
                        ],
                        shot=shot,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="auc",
                    state_name=state_name,
                    classes=1,
                    meta_data={"eval_func": binary_single_auc_func},
                )

            def make_pcba_data(
                split_name, state_name, shot=1, target_class=None
            ):
                return DataWithMeta(
                    GraphListHierFSDataset(
                        pcba_dataset,
                        pcba_dataset.label_text_feat,
                        pcba_dataset.prompt_edge_feat,
                        pcba_dataset.prompt_text_feat,
                        pcba_split[split_name],
                        single_prompt_edge=True,
                        walk_length=walk_length,
                        class_ind=pcba_dataset.y.view(len(pcba_dataset), -1)[
                            pcba_split[split_name]
                        ],
                        shot=shot,
                        target_class=target_class,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="auc",
                    state_name=state_name,
                    classes=1,
                    meta_data={"eval_func": binary_single_auc_func},
                )

            # def hiv_trim_class(embs, label):
            #     return label.view(1, -1), embs[0:1], label.view(1, -1)

            # def make_data(split_name, state_name, shot=5):
            #     return DataWithMeta(
            #         GraphListHierFixDataset(
            #             hiv_dataset,
            #             hiv_dataset.label_text_feat,
            #             hiv_dataset.prompt_edge_feat,
            #             hiv_dataset.prompt_text_feat,
            #             hiv_split[split_name],
            #             trim_class_func=hiv_trim_class,
            #             single_prompt_edge=True,
            #             walk_length=walk_length,
            #             class_ind=hiv_dataset.y.view(len(hiv_dataset), -1)[
            #                 hiv_split[split_name], 0:1
            #             ],
            #             shot=shot,
            #         ),
            #         batch_size,
            #         sample_size=sample_size,
            #         metric="auc",
            #         state_name=state_name,
            #         classes=1,
            #         meta_data={"eval_func": binary_single_auc_func},
            #     )
            hiv_val_data = [
                make_data("valid", "valid_hiv_" + str(i), i) for i in shots
            ]
            hiv_test_data = [
                make_data("test", "test_hiv_" + str(i), i) for i in shots
            ]

            pcba_val_data = [
                make_pcba_data(
                    "valid",
                    "valid_pcba_" + str(i) + "_" + str(j),
                    i,
                    torch.tensor([j]),
                )
                for i in shots
                for j in pcba_classes
            ]
            pcba_test_data = [
                make_pcba_data(
                    "test",
                    "test_pcba_" + str(i) + "_" + str(j),
                    i,
                    torch.tensor([j]),
                )
                for i in shots
                for j in pcba_classes
            ]

            split_data = {
                "train": [
                    GraphListHierFSDataset(
                        chembl_dataset,
                        chembl_dataset.label_text_feat,
                        chembl_dataset.prompt_edge_feat,
                        chembl_dataset.prompt_text_feat,
                        chembl_split["train"],
                        trim_class_func=trim_class,
                        single_prompt_edge=True,
                        walk_length=walk_length,
                        class_ind=chembl_classes,
                    )
                ],
                "test": hiv_test_data + pcba_test_data,
                "val": hiv_val_data + pcba_val_data,
            }
            constructed_data.append(split_data)
        if name == "molfewfull":
            chembl_dataset = CHEMBLPREOFADataset(
                "chemblpre", sentence_encoder=encoder
            )
            hiv_dataset = CHEMHIVOFADataset(
                "chemhiv", sentence_encoder=encoder
            )
            pcba_dataset = CHEMPCBAOFADataset(
                "chempcba", sentence_encoder=encoder
            )
            pcba_split = pcba_dataset.get_idx_split()
            hiv_split = hiv_dataset.get_idx_split()
            chembl_split = chembl_dataset.get_idx_split()
            chembl_classes = chembl_dataset.y.view(len(chembl_dataset), -1)[
                chembl_split["train"]
            ]

            def trim_class(embs, classes):
                valid_idx = classes == classes
                # valid_idx = torch.zeros_like(classes, dtype=torch.bool)
                return (
                    torch.tensor([[0]]),
                    embs[valid_idx.view(-1)].detach().clone(),
                    classes[:, valid_idx.view(-1)].detach().clone(),
                )

            def hiv_trim_class(embs, label):
                return label.view(1, -1), embs[0:1], label.view(1, -1)

            def make_data(split_name, state_name, shot=5):
                return DataWithMeta(
                    GraphListHierFixDataset(
                        hiv_dataset,
                        hiv_dataset.label_text_feat,
                        hiv_dataset.prompt_edge_feat,
                        hiv_dataset.prompt_text_feat,
                        hiv_split[split_name],
                        trim_class_func=hiv_trim_class,
                        single_prompt_edge=True,
                        walk_length=walk_length,
                        class_ind=hiv_dataset.y.view(len(hiv_dataset), -1)[
                            hiv_split[split_name], 0:1
                        ],
                        shot=1,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="auc",
                    state_name=state_name,
                    classes=2,
                    meta_data={"eval_func": binary_single_auc_func},
                )

            def make_pcba_data(split_name, state_name, shot=5):
                return DataWithMeta(
                    GraphListHierFixDataset(
                        pcba_dataset,
                        pcba_dataset.label_text_feat,
                        hiv_dataset.prompt_edge_feat,
                        pcba_dataset.prompt_text_feat,
                        pcba_split[split_name],
                        trim_class_func=None,
                        single_prompt_edge=True,
                        walk_length=walk_length,
                        class_ind=pcba_dataset.y.view(len(pcba_dataset), -1)[
                            pcba_split[split_name]
                        ],
                        shot=5,
                    ),
                    batch_size,
                    sample_size=sample_size,
                    metric="apr",
                    state_name=state_name,
                    classes=len(pcba_dataset.label_text_feat),
                    meta_data={"eval_func": binary_apr_func},
                )

            split_data = {
                "train": [
                    GraphListHierFSDataset(
                        chembl_dataset,
                        chembl_dataset.label_text_feat,
                        chembl_dataset.prompt_edge_feat,
                        chembl_dataset.prompt_text_feat,
                        chembl_split["train"],
                        trim_class_func=trim_class,
                        single_prompt_edge=True,
                        walk_length=walk_length,
                        class_ind=chembl_classes,
                        shot=2,
                    )
                ],
                "test": [
                    make_data("test", "test_molhiv"),
                    # make_data("train", "test_train_molhiv"),
                    # make_pcba_data("test", "test_pcba"),
                    # make_data("train", "test_train_molhiv"),
                ],
                "val": [
                    make_data("valid", "valid_molhiv"),
                    # make_pcba_data("valid", "valid_pcba"),
                ],
            }
            constructed_data.append(split_data)
    states = ["train", "test", "val"]
    collate_dataset = {}
    for s in states:
        dataset_list = []
        for ds in constructed_data:
            dataset_list += ds[s]
        collate_dataset[s] = dataset_list
    return collate_dataset


def main(params):
    encoder = SentenceEncoder("ST")

    if hasattr(params, "data_list"):
        e2e_data_list = params.data_list.split(",")
    else:
        e2e_data_list = [
            "coralink",
            "coranode",
            "pubmedlink",
            "pubmednode",
            "arxiv",
            "WN18RR",
            "FB15K237",
            "wikics",
            "chemblpre",
            "chempcba",
            "chemhiv",
        ]

    collate_dataset = data_construct(
        e2e_data_list,
        encoder,
        batch_size=params.eval_batch_size,
        sample_size=params.eval_sample_size,
        walk_length=params.rwpe,
    )

    out_dim = 768 + (params.rwpe if params.rwpe is not None else 0)
    # out_dim = 768

    def make_data(data, b_size, sample_size):
        return DataWithMeta(data, b_size, sample_size=sample_size)

    if hasattr(params, "d_multiple"):
        data_multiple = [float(a) for a in params.d_multiple.split(",")]
    else:
        data_multiple = [2, 2, 0.3, 2, 0.5, 0.4, 0.3, 2, 1, 2, 3]

    if hasattr(params, "d_min_ratio"):
        min_ratio = [float(a) for a in params.d_min_ratio.split(",")]
    else:
        min_ratio = [0.5, 0.5, 0.05, 1, 0.1, 0.1, 0.03, 1, 0.2, 0.2, 1]

    train_data = MultiDataset(
        collate_dataset["train"],
        dataset_multiple=data_multiple,
        # dataset_multiple=1,
        patience=3,
        window_size=5,
        min_ratio=min_ratio,
        # min_ratio=1,
    )

    text_dataset = {
        "train": make_data(
            train_data,
            params.batch_size,
            params.train_sample_size,
        ),
        "val": collate_dataset["val"],
        "test": collate_dataset["test"],
    }
    params.datamodule = DataModule(
        text_dataset, num_workers=params.num_workers
    )

    eval_data = text_dataset["val"] + text_dataset["test"]
    val_state = [dt.state_name for dt in text_dataset["val"]]
    test_state = [dt.state_name for dt in text_dataset["test"]]
    eval_state = val_state + test_state
    eval_metric = [dt.metric for dt in eval_data]
    eval_funcs = [dt.meta_data["eval_func"] for dt in eval_data]
    loss = torch.nn.BCEWithLogitsLoss()
    evlter = []
    for dt in eval_data:
        if dt.metric == "acc":
            evlter.append(Accuracy(task="multiclass", num_classes=dt.classes))
        elif dt.metric == "auc":
            evlter.append(AUROC(task="binary"))
        elif dt.metric == "apr":
            evlter.append(MultiApr(num_labels=dt.classes))
        elif dt.metric == "aucmulti":
            evlter.append(MultiAuc(num_labels=dt.classes))
    metrics = EvalKit(
        eval_metric,
        evlter,
        loss,
        eval_funcs,
        flat_binary_func,
        eval_mode="max",
        exp_prefix="",
        eval_state=eval_state,
        val_monitor_state=val_state[0],
        test_monitor_state=test_state[0],
    )
    # gnn = PyGGIN(params.num_layers, 768, 768)
    # gnn = PyGRGCN(params.num_layers, 3, 768, 768)
    # gnn = PyGGINE(params.num_layers, 768, 768, 768)
    gnn = PyGRGCNEdge(
        params.num_layers,
        5,
        out_dim,
        out_dim,
        drop_ratio=params.dropout,
        JK=params.JK,
    )
    bin_model = BinGraphAttModel if params.JK == "none" else BinGraphModel
    model = bin_model(gnn, out_dim, 1, add_rwpe=params.rwpe)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=params.l2
    )
    lr_scheduler = {
        "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.5),
        "interval": "epoch",
        "frequency": 1,
    }

    exp_config = ExpConfig(
        "",
        optimizer,
        dataset_callback=train_data.update,
        lr_scheduler=lr_scheduler,
    )
    exp_config.val_state_name = val_state
    exp_config.test_state_name = test_state

    pred_model = GraphPredLightning(exp_config, model, metrics)

    wandb_logger = WandbLogger(
        project=params.log_project,
        name=params.exp_name,
        save_dir=params.exp_dir,
        offline=params.offline_log,
    )

    val_res, test_res = lightning_fit(
        wandb_logger,
        pred_model,
        params.datamodule,
        metrics,
        params.num_epochs,
        save_model=False,
        load_best=False,
        reload_freq=1,
        test_rep=params.test_rep
        # profiler="simple",
        # accelerator="cpu",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rl")

    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=3)
    parser.add_argument("--q_query", type=int, default=3)
    parser.add_argument(
        "--fs_task_num",
        type=int,
        default=5,
        help="Number of tasks for few-shot training.",
    )

    params = parser.parse_args()
    configs = []
    configs.append(
        load_yaml(
            os.path.join(
                os.path.dirname(__file__), "configs", "default_config.yaml"
            )
        )
    )
    print(configs)
    # Add for few-shot parameters
    configs.append(params.__dict__)

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)
    setup_exp(mod_params)

    params = SimpleNamespace(**mod_params)
    set_random_seed(params.seed)

    torch.set_float32_matmul_precision("high")
    params.log_project = "full_cdm"
    main(params)
