import torch
import torch_geometric as pyg
import json
import numpy as np
import copy

from data.KG.gen_data import KGOFADataset
from data.chemmol.gen_data import MolOFADataset
from data.single_graph.gen_data import SingleGraphOFADataset

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
    SubgraphNopromptLinkDataset,
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
    flat_auc,
)

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

name2dataset = {
    "arxiv": SingleGraphOFADataset,
    "Cora": SingleGraphOFADataset,
    "Pubmed": SingleGraphOFADataset,
    "WN18RR": KGOFADataset,
    "FB15K237": KGOFADataset,
    "wikics": SingleGraphOFADataset,
    "chemblpre": MolOFADataset,
    "chempcba": MolOFADataset,
    "chemhiv": MolOFADataset,
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
        meta_data={"eval_func": eval_func, "eval_mode": kwargs["eval_mode"]},
    )


def ConstructNodeCls(
        dataset, split, split_name, prompt_feats, to_bin_cls_func, global_data, **kwargs
):
    text_g = dataset.data

    return SubgraphHierDataset(
        text_g,
        prompt_feats["class_node_text_feat"],
        prompt_feats["prompt_edge_text_feat"],
        prompt_feats["noi_node_text_feat"],
        split[split_name],
        to_undirected=True,
        process_label_func=to_bin_cls_func,
        **kwargs,
    )


def ConstructNodeNopromptCls(
        dataset, split, split_name, to_bin_cls_func, global_data, **kwargs
):
    text_g = dataset.data

    return SubgraphNopromptDataset(
        text_g,
        text_g.label_text_feat,
        split[split_name],
        to_undirected=True,
        process_label_func=to_bin_cls_func,
    )


def ConstructLinkCls(
        dataset, split, split_name, prompt_feats, to_bin_cls_func, global_data, **kwargs
):
    text_g = dataset.data
    edges = text_g.edge_index
    train_graph = global_data

    return SubgraphLinkHierDataset(
        train_graph,
        prompt_feats["class_node_text_feat"],
        prompt_feats["prompt_edge_text_feat"],
        prompt_feats["noi_node_text_feat"],
        edges.T[split[split_name]].numpy(),
        to_undirected=True,
        hop=3,
        process_label_func=to_bin_cls_func,
        **kwargs,
    )


def ConstructLinkNopromptCls(
        dataset, split, split_name, to_bin_cls_func, **kwargs
):
    text_g = dataset.data
    edges = text_g.edge_index
    train_graph = kwargs["global_data"]

    return SubgraphNopromptLinkDataset(
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


def ConstructKG(dataset, split, split_name, prompt_feats, to_bin_cls_func, global_data, **kwargs):
    text_g = dataset.data

    return SubgraphKGHierDataset(
        text_g,
        prompt_feats["class_node_text_feat"],
        prompt_feats["prompt_edge_text_feat"],
        prompt_feats["noi_node_text_feat"],
        split[split_name],
        to_undirected=True,
        hop=2,
        process_label_func=to_bin_cls_func,
        **kwargs,
    )


def ConstructMolCls(
        dataset, split, split_name, prompt_feats, to_bin_cls_func, global_data, **kwargs
):
    return GraphListHierDataset(
        dataset,
        prompt_feats["class_node_text_feat"],
        prompt_feats["prompt_edge_text_feat"],
        prompt_feats["noi_node_text_feat"],
        split[split_name],
        process_label_func=to_bin_cls_func,
        single_prompt_edge=True,
        **kwargs,
    )


def ConstructMolNopromptCls(
        dataset, split, split_name, to_bin_cls_func, **kwargs
):
    return GraphListNopromptDataset(
        dataset,
        dataset.label_text_feat,
        dataset.prompt_edge_feat,
        split[split_name],
        process_label_func=to_bin_cls_func,
        single_prompt_edge=True,
        walk_length=kwargs["walk_length"],
    )


def ConstructNCFSZS(
        dataset, data_manager, n, k, split_name, config, state_name=None, eval_metric=None, eval_func=None,
        train_flag=False, adj=None, total_task_num=50, undirected_flag=True, **kwargs
):
    if config["class_emb_flag"]:
        class_emb = dataset.label_text_feat
    else:
        class_emb = dataset.prompt_text_feat.repeat(
            len(dataset.label_text_feat), 1
        )
    random_flag = config["random_flag"] if split_name == "train" else None
    split_name = config["mode"][split_name]
    data_class = FewShotNCDataset if kwargs["k_shot"] > 0 else ZeroShotNCDataset
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
        dataset, data_manager, n, k, split_name, config, state_name=None, eval_metric=None, eval_func=None,
        train_flag=False, adj=None, total_task_num=50, undirected_flag=True, **kwargs
):
    if config["class_emb_flag"]:
        class_emb = dataset.edge_label_feat
    else:
        class_emb = dataset.prompt_text_feat.repeat(
            len(dataset.edge_label_feat), 1
        )
    random_flag = config["random_flag"] if split_name == "train" else None
    split_name = config["mode"][split_name]
    data_class = FewShotKGDataset if kwargs["k_shot"] > 0 else ZeroShotKGDataset
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


def ConstructGCFSZS(
        dataset, split, split_name, n, k, config, state_name=None, eval_metric=None, eval_func=None, train_flag=False,
        batch_size=None, **kwargs
):
    data_class = GraphListHierFSDataset if kwargs["k_shot"] > 0 else GraphListHierDataset
    ofa_data = data_class(
        graphs=dataset,
        class_embs=dataset.label_text_feat,
        prompt_edge_feat=dataset.prompt_edge_feat,
        prompt_text_feat=dataset.prompt_text_feat,
        data_idx=split[split_name],
        process_label_func=globals()[config["process_label_func"]],
        single_prompt_edge=True,
        walk_length=kwargs["walk_length"],
        class_ind=dataset.y.view(len(dataset), -1)[split[split_name], 0:1] if kwargs["k_shot"] > 0 else None,
        shot=k,
        target_class=n,
    )

    if train_flag:
        return ofa_data
    else:
        return DataWithMeta(
            ofa_data,
            batch_size=batch_size,
            sample_size=-1,
            metric=eval_metric,
            state_name=state_name,
            classes=kwargs["classes"],
            meta_data={"eval_func": eval_func}
        )


def process_pth_label(embs, label):
    binary_rep = torch.zeros((1, len(embs)))
    binary_rep[0, label.squeeze().to(torch.long)] = 1
    return label.view(1, -1).to(torch.long), embs, binary_rep


def process_reverse_binary_label(embs, label):
    binary_rep = torch.zeros((1, len(embs)))
    binary_rep[0, label.squeeze().to(torch.long)] = 1
    embs = embs[[1, 0]]
    return label.view(1, -1).to(torch.long), embs, binary_rep


def process_multi_label(embs, label):
    valid_idx = label == label
    # valid_idx = torch.zeros_like(classes, dtype=torch.bool)
    return (
        torch.tensor([[0]]),
        embs[valid_idx.view(-1)].detach().clone(),
        label[:, valid_idx.view(-1)].detach().clone(),
    )


def process_positive_negative_multi_label(embs, label):
    valid_idx = label == label
    label = label[:, valid_idx.view(-1)].detach().clone()
    valid_idx = valid_idx.repeat(1, 2)
    label = torch.cat([label, 1 - label], dim=-1)

    return (
        torch.tensor([[0]]),
        embs[valid_idx.view(-1)].detach().clone(),
        label,
    )


def eval_process_label(embs, classes):
    return (
        torch.tensor([[0]]),
        embs,
        classes,
    )


def process_label_positive_only(embs, label):
    return torch.tensor([[0]]), embs[:len(label.view(-1))], label


def process_int_label(embs, label):
    binary_rep = torch.zeros((1, len(embs)))
    binary_rep[0, label] = 1
    return torch.tensor([label]).view(1, -1), embs, binary_rep


def hiv_trim_class(embs, label):
    one_hot_label = torch.nn.functional.one_hot(
        label.to(torch.long), num_classes=2
    )
    return label, embs, one_hot_label


def hiv_zs_class(embs, label):
    # one_hot_label = torch.nn.functional.one_hot(
    #     label.to(torch.long), num_classes=2
    # )
    return label, embs[0:1], label


def gen_can(n_class, label, size):
    can = torch.randint(n_class, size)
    mask = torch.rand(size) > 0.75
    can[mask] = label.view(-1)
    return can


def process_logic_label(embs, label):
    num_class = int(np.sqrt(len(embs) / 2))
    can = gen_can(num_class, label, (4, 2))
    or_label = ((can == label.view(-1)).sum(-1) > 0).to(torch.int)
    or_feat = embs[can[:, 0] * num_class + can[:, 1]]

    can = gen_can(num_class, label, (4, 2))
    and_label = ((can == label.view(-1)).sum(-1) == 0).to(torch.int)
    and_feat = embs[can[:, 0] * num_class + can[:, 1] + num_class ** 2]
    new_class_emb = torch.cat([or_feat, and_feat], dim=0)
    new_binary_rep = torch.cat([or_label, and_label]).view(1, -1)
    if isinstance(label, int):
        label = torch.tensor(label)
    return label.view(1, -1).to(torch.long), new_class_emb, new_binary_rep


none_process_label = None


class UnifiedTaskConstructor:
    def __init__(self, tasks, encoder, task_config_lookup, data_config_lookup, root="cache_data", batch_size=256,
                 sample_size=-1):
        self.root = root
        self.tasks = tasks
        self.encoder = encoder
        self.task_config_lookup = task_config_lookup
        self.data_config_lookup = data_config_lookup
        self.batch_size = batch_size
        self.sample_size = sample_size
        with open("data/low_resource_split.json", "r") as f:
            self.lr_class_split = json.load(f)

        self.dataset = {}  # keyed by base dataset names e.g. cora, pubmed and not cora-link
        self.dataset_split = {}  # keyed by dataset names and task level e.g. cora_e2e_link
        self.preprocess_storage = {}  # keyed by dataset names and task level e.g. cora_e2e_link
        self.datamanager = {}
        self.edges = {}
        self.datasets = {"train": [], "valid": [],
                         "test": []}  # train a list of Dataset, valid/test a list of DataWithMeta
        self.stage_names = {"train": [], "valid": [], "test": []}

    def construct_exp(self):
        val_task_index_lst = []
        val_pool_mode = []
        for task in self.tasks:
            config = self.task_config_lookup[task]
            config = copy.deepcopy(config)
            val_task_index_lst.append(self.construct_task(config))
            val_pool_mode.append(config["eval_pool_mode"])
        return val_task_index_lst, val_pool_mode

    def construct_task(self, config):
        val_task_index = []
        for stage_config in config["eval_set_constructs"]:
            if "dataset" not in stage_config:
                stage_config["dataset"] = config["dataset"]
            dataset_name = stage_config["dataset"]

            assert dataset_name in self.data_config_lookup

            dataset_config = self.data_config_lookup[dataset_name]

            stage_ind = self.add_dataset(stage_config, dataset_config)

            if stage_config["stage"] == "valid":
                val_task_index.append(stage_ind)
        return val_task_index

    def get_split_key(self, dataset_config):
        return dataset_config["dataset_name"] + "_" + dataset_config["task_level"]

    def get_stage_name(self, stage_config, dataset_config):
        return "_".join([self.get_split_key(dataset_config), stage_config["stage"], stage_config["split_name"]])

    def get_ofa_data(self, dataset_config):
        dataset_name = dataset_config["dataset_name"]
        if dataset_name not in self.dataset:
            self.dataset[dataset_name] = name2dataset[dataset_name](
                dataset_name, root=self.root, encoder=self.encoder
            )
        return self.dataset[dataset_name]

    def get_data_split(self, dataset_config):
        split_key = self.get_split_key(dataset_config)
        if split_key not in self.dataset_split:
            dataset_splitter = dataset_config.get("dataset_splitter")
            split = globals()[dataset_splitter](
                self.dataset[dataset_config["dataset_name"]]) if dataset_splitter else None
            self.dataset_split[split_key] = split
        return self.dataset_split[split_key]

    def get_global_data(self, dataset_config):
        split_key = self.get_split_key(dataset_config)
        if split_key not in self.preprocess_storage:
            preprocessor = dataset_config.get("preprocess")
            global_data = globals()[preprocessor](self.dataset[dataset_config["dataset_name"]],
                                                  self.dataset_split[split_key]) if preprocessor else None
            self.preprocess_storage[split_key] = global_data
        return self.preprocess_storage[split_key]

    def add_dataset(self, stage_config, dataset_config):
        data = self.get_ofa_data(dataset_config)
        split = self.get_data_split(dataset_config)
        stage_name = self.get_stage_name(stage_config, dataset_config)
        if stage_config["stage"] != "train" and stage_name in self.stage_names[stage_config["stage"]]:
            return self.stage_names[stage_config["stage"]].index(stage_name)
        global_data = self.get_global_data(dataset_config)
        prompt_feats = data.get_prompt_text_feat(dataset_config["task_level"])
        data = globals()[dataset_config["construct"]](
            dataset=data,
            split=split,
            split_name=stage_config["split_name"],
            prompt_feats=prompt_feats,
            to_bin_cls_func=globals()[dataset_config["process_label_func"]] if dataset_config.get(
                "process_label_func") else None,
            global_data=global_data,
            **dataset_config["args"],
        )
        if stage_config["stage"] == "train":
            self.datasets[stage_config["stage"]].append(data)
        else:
            eval_data = make_data(
                stage_config["dataset"],
                data,
                stage_config["split_name"],
                dataset_config["eval_metric"],
                globals()[dataset_config["eval_func"]],
                dataset_config["num_classes"],
                batch_size=self.batch_size,
                sample_size=self.sample_size,
                eval_mode=dataset_config["eval_mode"]
            )
            self.datasets[stage_config["stage"]].append(eval_data)
        self.stage_names[stage_config["stage"]].append(stage_name)
        return self.stage_names[stage_config["stage"]].index(stage_name)

    def make_train_data(self, multiple, min_ratio, data_val_index=None):
        train_data = MultiDataset(
            self.datasets["train"],
            data_val_index=data_val_index,
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
            "val": self.datasets["valid"],
            "test": self.datasets["test"],
        }
        return text_dataset


def get_eval_data(self, task, config, data, split, global_data, g, args):
    if not config["train_only"]:
        for eval_construct_config in config["eval_set_constructs"]:
            if "args" in eval_construct_config:
                eval_args = eval_construct_config["args"]
            else:
                eval_args = args

            if "construct" in eval_construct_config:
                construct = globals()[eval_construct_config["construct"]]
            else:
                construct = globals()[config["construct"]]

            if "lr" in config["task_level"]:
                eval_data = self.get_lr_eval_data(construct, data, g, config, split, eval_construct_config,
                                                  eval_args)
            else:
                eval_data = self.get_e2e_eval_data(construct, data, task, split, global_data, config,
                                                   eval_construct_config, eval_args)

            if eval_construct_config["stage"] == "valid":
                self.valid_dm_set += eval_data
            else:
                self.test_dm_set += eval_data


def get_lr_eval_data(self, construct, data, g, config, split, eval_construct_config, eval_args):
    eval_data = [
        construct(
            dataset=g if g else self.dataset[data],
            data_manager=self.datamanager.get(data),
            n=n,
            k=k,
            split_name=eval_construct_config["stage"],
            config=config,
            split=split,
            state_name=f'{eval_construct_config["stage"]}_fs{n}_{k}_{data}' if n else f'{eval_construct_config["stage"]}_fs{k}_{data}',
            eval_metric=config["eval_metric"],
            eval_func=globals()[config["eval_func"]],
            batch_size=self.batch_size,
            **eval_args,
        )
        for n in eval_args["val_n"]
        for k in eval_args["val_k"]
    ]
    return eval_data


def get_e2e_eval_data(self, construct, data, task, split, global_data, config, eval_construct_config, eval_args):
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
        task,
        eval_data,
        eval_construct_config["split_name"],
        config["eval_metric"],
        globals()[config["eval_func"]],
        config["num_classes"],
        batch_size=self.batch_size,
        sample_size=self.sample_size,
    )
    return [dm_data]


def get_graph(self, data, task_level, class_split_lst=None):
    # preprocess graph/edges for few-shot and zero-shot tasks
    dataset = self.dataset[data]
    g = dataset.data
    g.x = g.x_text_feat

    if "link" in task_level:
        if data not in self.edges:
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
