import torch
import torch_geometric as pyg

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
)

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


def ArxivNodeClsTask(name, dataset, **kwargs):
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
                walk_length=kwargs["walk_length"],
            ),
            kwargs["batch_size"],
            sample_size=kwargs["sample_size"],
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
                walk_length=kwargs["walk_length"],
            )
        ],
        "test": [
            make_data(2, "test_arxiv"),
            make_data(0, "test_train_arxiv"),
        ],
        "val": [make_data(1, "valid_arxiv")],
    }
    return split_data


def CiteNodeClsTask(name, dataset, **kwargs):
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
                walk_length=kwargs["walk_length"],
            ),
            kwargs["batch_size"],
            sample_size=kwargs["sample_size"],
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
                walk_length=kwargs["walk_length"],
            )
        ],
        "test": [
            make_data("test", "test_" + name),
            make_data("train", "test_train_" + name),
        ],
        "val": [make_data("val", "valid_" + name)],
    }
    return split_data


def CiteLinkTask(name, dataset, **kwargs):
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
                walk_length=kwargs["walk_length"],
            ),
            kwargs["batch_size"],
            sample_size=kwargs["sample_size"],
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
                walk_length=kwargs["walk_length"],
            )
        ],
        "test": [
            make_data("test", "test_" + name, remove_edge=False),
            make_data("train", "test_train_" + name, remove_edge=True),
        ],
        "val": [make_data("val", "valid_" + name, remove_edge=False)],
    }
    return split_data


def KGLinkTask(name, dataset, **kwargs):
    text_g = dataset.data
    text_g.x = text_g.x_text_feat
    text_g.prompt_edge_feat = dataset.prompt_edge_feat
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
                walk_length=kwargs["walk_length"],
            ),
            kwargs["batch_size"],
            sample_size=kwargs["sample_size"],
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
                walk_length=kwargs["walk_length"],
            )
        ],
        "test": [
            make_data("test", "test_" + name, remove_edge=False),
            make_data("train", "test_train_" + name, remove_edge=True),
        ],
        "val": [make_data("valid", "valid_" + name, remove_edge=False)],
    }
    return split_data


def WikiNodeClsTask(name, dataset, **kwargs):
    text_g = dataset.data
    text_g.x = text_g.x_text_feat
    text_g.prompt_edge_feat = dataset.prompt_edge_feat
    wiki_split_idx = 0
    split = [
        torch.where(text_g.train_mask[:, wiki_split_idx])[0].numpy(),
        torch.where(text_g.val_mask[:, wiki_split_idx])[0].numpy(),
        torch.where(text_g.test_mask)[0].numpy(),
    ]

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
                walk_length=kwargs["walk_length"],
            ),
            kwargs["batch_size"],
            sample_size=kwargs["sample_size"],
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
                walk_length=kwargs["walk_length"],
            )
        ],
        "test": [
            make_data("test", "test_" + name),
            make_data("train", "test_train_" + name),
        ],
        "val": [make_data("val", "valid_" + name)],
    }
    return split_data


def MOLMultiClsTask(name, dataset, **kwargs):
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
                walk_length=kwargs["walk_length"],
            ),
            kwargs["batch_size"],
            sample_size=kwargs["sample_size"],
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
                walk_length=kwargs["walk_length"],
            )
        ],
        "test": [
            make_data("test", "test_" + name),
            make_data("train", "test_train_" + name),
        ],
        "val": [make_data("valid", "valid_" + name)],
    }
    return split_data


def MOLClsTask(name, dataset, **kwargs):
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
        one_hot_label = torch.nn.functional.one_hot(label, num_classes=2)
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
                walk_length=kwargs["walk_length"],
            ),
            kwargs["batch_size"],
            sample_size=kwargs["sample_size"],
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
                walk_length=kwargs["walk_length"],
            )
        ],
        "test": [
            make_data("test", "test_" + name),
            make_data("train", "test_train_" + name),
        ],
        "val": [make_data("valid", "valid_" + name)],
    }
    return split_data


def MOLZeroClsTask(name, dataset, **kwargs):
    chembl_dataset = dataset[0]
    hiv_dataset = dataset[1]
    pcba_dataset = dataset[2]
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
                walk_length=kwargs["walk_length"],
            ),
            kwargs["batch_size"],
            sample_size=kwargs["sample_size"],
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
                walk_length=kwargs["walk_length"],
            ),
            kwargs["batch_size"],
            sample_size=kwargs["sample_size"],
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
                walk_length=kwargs["walk_length"],
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
    return split_data


def MOLFewClsTask(name, dataset, **kwargs):
    chembl_dataset = dataset[0]
    hiv_dataset = dataset[1]
    pcba_dataset = dataset[2]
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
                walk_length=kwargs["walk_length"],
                class_ind=hiv_dataset.y.view(len(hiv_dataset), -1)[
                    hiv_split[split_name], 0:1
                ],
                shot=shot,
            ),
            kwargs["batch_size"],
            sample_size=kwargs["sample_size"],
            metric="auc",
            state_name=state_name,
            classes=1,
            meta_data={"eval_func": binary_single_auc_func},
        )

    def make_pcba_data(split_name, state_name, shot=1, target_class=None):
        return DataWithMeta(
            GraphListHierFSDataset(
                pcba_dataset,
                pcba_dataset.label_text_feat,
                pcba_dataset.prompt_edge_feat,
                pcba_dataset.prompt_text_feat,
                pcba_split[split_name],
                single_prompt_edge=True,
                walk_length=kwargs["walk_length"],
                class_ind=pcba_dataset.y.view(len(pcba_dataset), -1)[
                    pcba_split[split_name]
                ],
                shot=shot,
                target_class=target_class,
            ),
            kwargs["batch_size"],
            sample_size=kwargs["sample_size"],
            metric="auc",
            state_name=state_name,
            classes=1,
            meta_data={"eval_func": binary_single_auc_func},
        )

    hiv_val_data = [
        make_data("valid", "valid_hiv_" + str(i), i) for i in shots
    ]
    hiv_test_data = [make_data("test", "test_hiv_" + str(i), i) for i in shots]

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
                walk_length=kwargs["walk_length"],
                class_ind=chembl_classes,
            )
        ],
        "test": hiv_test_data + pcba_test_data,
        "val": hiv_val_data + pcba_val_data,
    }
    return split_data


taskname2func = {
    "arxiv": ArxivNodeClsTask,
    "cora_node": CiteNodeClsTask,
    "pubmed_node": CiteNodeClsTask,
    "cora_link": CiteLinkTask,
    "pubmed_link": CiteLinkTask,
    "WN18RR": KGLinkTask,
    "FB15K237": KGLinkTask,
    "wikics": WikiNodeClsTask,
    "chemblpre": MOLMultiClsTask,
    "chempcba": MOLMultiClsTask,
    "chemhiv": MOLClsTask,
    "molzero": MOLZeroClsTask,
}


class TaskConstructor:
    def __init__(self, tasks, encoder):
        self.tasks = tasks
        self.dataset = {}

        for task in self.tasks:
            data = task.split("_")[0]
            if data not in self.dataset and data in name2dataset:
                self.dataset[data] = name2dataset[data](data, encoder)
