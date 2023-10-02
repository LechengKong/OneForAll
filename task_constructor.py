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
        "val": edge_perm[train_offset:val_offset],
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


def ConstructNodeClsTrain(name, dataset, split, split_name, **kwargs):
    text_g = dataset.data
    text_g.x = text_g.x_text_feat
    text_g.prompt_edge_feat = dataset.prompt_edge_feat

    def trim_class(label, num_class):
        binary_rep = torch.zeros((1, num_class))
        binary_rep[0, label] = 1
        return label.view(1, -1), binary_rep

    return SubgraphHierDataset(
        text_g,
        text_g.label_text_feat,
        split[split_name],
        prompt_feat=text_g.prompt_text_feat,
        to_undirected=True,
        trim_class_func=trim_class,
        walk_length=kwargs["walk_length"],
    )


def ConstructLinkTrain(name, dataset, split, **kwargs):
    text_g = dataset.data
    text_g.x = text_g.x_text_feat
    text_g.prompt_edge_feat = dataset.prompt_edge_feat
    edges = text_g.edge_index
    graph_dict = text_g.to_dict()
    graph_dict["edge_index"] = edges[:, split["train"]]
    train_graph = pyg.data.Data(**graph_dict)

    def trim_class(label, num_class):
        binary_rep = torch.zeros((1, num_class))
        binary_rep[0, label] = 1
        return torch.tensor([label]).view(1, -1), binary_rep

    return SubgraphLinkHierDataset(
        train_graph,
        train_graph.edge_label_feat,
        edges.T[split["train"]].numpy(),
        prompt_feat=train_graph.prompt_node_edge_feat,
        to_undirected=True,
        hop=3,
        remove_edge=True,
        trim_class_func=trim_class,
        walk_length=kwargs["walk_length"],
    )


def ConstructKGTrain(name, dataset, split, **kwargs):
    text_g = dataset.data
    text_g.x = text_g.x_text_feat
    text_g.prompt_edge_feat = dataset.prompt_edge_feat

    def trim_class(label, num_class):
        binary_rep = torch.zeros((1, num_class))
        binary_rep[0, label] = 1
        return torch.tensor([label]).view(1, -1), binary_rep

    return SubgraphKGHierDataset(
        text_g,
        text_g.edge_label_feat,
        split["train"],
        prompt_feat=text_g.prompt_text_feat,
        to_undirected=True,
        hop=2,
        remove_edge=True,
        trim_class_func=trim_class,
        walk_length=kwargs["walk_length"],
    )


def ConstructMolMultiClsTrain(name, dataset, split, **kwargs):
    def trim_class(embs, classes):
        valid_idx = classes == classes
        # valid_idx = torch.zeros_like(classes, dtype=torch.bool)
        return (
            torch.tensor([[0]]),
            embs[valid_idx.view(-1)].detach().clone(),
            classes[:, valid_idx.view(-1)].detach().clone(),
        )

    return GraphListHierDataset(
        dataset,
        dataset.label_text_feat,
        dataset.prompt_edge_feat,
        dataset.prompt_text_feat,
        split["train"],
        trim_class_func=trim_class,
        single_prompt_edge=True,
        walk_length=kwargs["walk_length"],
    )


def ConstructMolTrain(name, dataset, split, **kwargs):
    def trim_class(embs, label):
        label = label.to(torch.long)
        one_hot_label = torch.nn.functional.one_hot(label, num_classes=2)
        return label, embs, one_hot_label

    return GraphListHierDataset(
        dataset,
        dataset.label_text_feat,
        dataset.prompt_edge_feat,
        dataset.prompt_text_feat,
        split["train"],
        trim_class_func=trim_class,
        single_prompt_edge=True,
        walk_length=kwargs["walk_length"],
    )


def ConstructMolFSTrain(name, dataset, split, **kwargs):
    classes = dataset.y.view(len(dataset), -1)[split["train"]]

    def trim_class(embs, classes):
        valid_idx = classes == classes
        # valid_idx = torch.zeros_like(classes, dtype=torch.bool)
        return (
            torch.tensor([[0]]),
            embs[valid_idx.view(-1)].detach().clone(),
            classes[:, valid_idx.view(-1)].detach().clone(),
        )

    return GraphListHierFSDataset(
        dataset,
        dataset.label_text_feat,
        dataset.prompt_edge_feat,
        dataset.prompt_text_feat,
        split["train"],
        trim_class_func=trim_class,
        single_prompt_edge=True,
        walk_length=kwargs["walk_length"],
        class_ind=classes,
    )


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
