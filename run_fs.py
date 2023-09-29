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
from models.model import BinGraphModel, BinGraphAttModel
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
    classification_single_func,
)
from data.arxiv.gen_data import ArxivOFADataset
from data.Cora.gen_data import CoraOFADataset
from data.WN18RR.gen_data import WN18RROFADataset
from data.FB15K237.gen_data import FB15K237OFADataset
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
    wiki_train_g = None
    for data_idx, name in enumerate(data_names):
        print("loading: ", name)
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
                    sample_size=1000,
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
                    sample_size=1000,
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

        if name.startswith("fs_"):

            (
                [n_way, min_n],
                [k_shot, min_k],
                q_query,
                fs_task_num,
                class_emb_flag,
            ) = fs_setting[name]

            def make_data(
                g,
                data_manager,
                split_name: int,
                state_name,
                undirected_flag,
                adj=None,
                class_emb_flag=False,
                train_flag=False,
                random_flag=False,
                min_n=None,
                min_k=None,
            ):
                if class_emb_flag:
                    class_emb = g.label_text_feat
                else:
                    class_emb = g.prompt_text_feat.repeat(
                        len(g.label_text_feat), 1
                    )
                if k_shot > 0:
                    data_class = FewShotNCDataset
                else:
                    data_class = ZeroShotNCDataset
                dataset = data_class(
                    pyg_graph=g,
                    class_emb=class_emb,
                    data_idx=torch.zeros(50),
                    n_way=n_way,
                    k_shot=k_shot,
                    q_query=q_query,
                    datamanager=data_manager,
                    mode=split_name,
                    hop=2,
                    prompt_feat=g.prompt_text_feat,
                    to_undirected=undirected_flag,
                    adj=adj,
                    single_prompt_edge=True,
                    random_flag=random_flag,
                    min_n=min_n,
                    min_k=min_k,
                )
                if train_flag:
                    return dataset
                return DataWithMeta(
                    dataset,
                    batch_size=fs_task_num,
                    sample_size=-1,
                    metric="acc",
                    state_name=state_name,
                    classes=min_n,
                    meta_data={"eval_func": classification_single_func},
                )

            if "arxiv" in name:
                dataset = ArxivOFADataset("arxiv", sentence_encoder=encoder)
                text_g = dataset.data
                text_g.x = text_g.x_text_feat
                text_g.prompt_edge_feat = dataset.prompt_edge_feat
                data_manager = FewShotDataManager(
                    text_g,
                    n_way,
                    k_shot,
                    q_query,
                )
                # only use cora for test
                dataset_cora = CoraOFADataset("cora", sentence_encoder=encoder)
                g_cora = dataset_cora.data
                g_cora.x = dataset_cora.x_text_feat
                g_cora.prompt_edge_feat = dataset_cora.prompt_edge_feat
                g_cora.prompt_text_feat = g_cora.prompt_node_feat
                cora_fsdm = FewShotDataManager(
                    g_cora, n_way, k_shot, q_query, class_split_ratio=[7, 0, 0]
                )

                if k_shot > 0:
                    k_shot_lst = [1, 3, 5]
                else:
                    k_shot_lst = [0]

                train_data = [
                    make_data(
                        text_g,
                        data_manager,
                        0,
                        f"train_fs_arxiv",
                        True,
                        class_emb_flag=class_emb_flag,
                        train_flag=True,
                        random_flag=True,
                        min_n=min_n,
                        min_k=min_k,
                    )
                ]
                valid_data = [
                    make_data(
                        g[0],
                        g[1],
                        g[2],
                        f"val_fs{n}{k}_{g[3]}",
                        True,
                        class_emb_flag=class_emb_flag,
                        min_n=n,
                        min_k=k,
                    )
                    for g in [
                        [text_g, data_manager, 1, "arxiv"],
                        [g_cora, cora_fsdm, 0, "cora"],
                    ]
                    for n in [5, 3]
                    for k in k_shot_lst
                ]
                test_data = [
                    make_data(
                        g[0],
                        g[1],
                        g[2],
                        f"test_fs{n}{k}_{g[3]}",
                        True,
                        class_emb_flag=class_emb_flag,
                        min_n=n,
                        min_k=k,
                    )
                    for g in [
                        [text_g, data_manager, 2, "arxiv"],
                        [g_cora, cora_fsdm, 0, "cora"],
                    ]
                    for n in [5, 3]
                    for k in k_shot_lst
                ]

                split_data = {
                    "train": train_data,
                    "val": valid_data,
                    "test": test_data,
                }
                constructed_data.append(split_data)

        if name.startswith("fskg_"):
            (
                [n_way, min_n],
                [k_shot, min_k],
                q_query,
                fs_task_num,
                class_emb_flag,
            ) = fs_setting[name]

            class_split_lst = None
            class_split_ratio = None

            dataset = FB15K237OFADataset("FB15K237", sentence_encoder=encoder)
            train_g = dataset.data
            train_g.x = train_g.x_text_feat
            train_g.prompt_edge_feat = dataset.prompt_edge_feat
            converted_triplet = dataset.get_idx_split()
            edges = torch.cat(
                [
                    torch.tensor(converted_triplet["train"][0]).T,
                    torch.tensor(converted_triplet["valid"][0]).T,
                    torch.tensor(converted_triplet["test"][0]).T,
                ],
                dim=-1,
            )
            train_g.edge_index = edges
            edge_labels = torch.cat(
                [
                    torch.tensor(converted_triplet["train"][1]),
                    torch.tensor(converted_triplet["valid"][1]),
                    torch.tensor(converted_triplet["test"][1]),
                ]
            )
            train_g.y = edge_labels
            train_adj = None
            train_datamanager = FewShotDataManager(
                train_g, n_way, k_shot, q_query
            )
            TRAIN_LABELS = [
                44,
                109,
                9,
                144,
                106,
                73,
                172,
                52,
                211,
                113,
                226,
                191,
                29,
                148,
                111,
                98,
                21,
                186,
                228,
                119,
                185,
                66,
                199,
                112,
                174,
                72,
                207,
                85,
                200,
                115,
                84,
                60,
                125,
                61,
                12,
                99,
                149,
                27,
                78,
                181,
                10,
                128,
                30,
                42,
                45,
                100,
                89,
                4,
                180,
                161,
                230,
                5,
                130,
                192,
                3,
                15,
                175,
                131,
                156,
                145,
                190,
                110,
                219,
                2,
                222,
                203,
                79,
                122,
                81,
                47,
                206,
                164,
                176,
                38,
                13,
                77,
                223,
                64,
                123,
                0,
                146,
                105,
                193,
                37,
                157,
                153,
                82,
                34,
                124,
                187,
                197,
                103,
                201,
                232,
                32,
                215,
                165,
                104,
                48,
                171,
                177,
                76,
                120,
                196,
                132,
                209,
                135,
                19,
                212,
                213,
                208,
                169,
                133,
                65,
                170,
                235,
                46,
                51,
                121,
                204,
                134,
                167,
                126,
                33,
                138,
                162,
                95,
                63,
                229,
                184,
                36,
                182,
                101,
                127,
                152,
                140,
                54,
                16,
                155,
                102,
                80,
                220,
            ]

            VAL_LABELS = [
                168,
                14,
                83,
                205,
                142,
                69,
                236,
                118,
                159,
                231,
                136,
                43,
                18,
                68,
                53,
                90,
                94,
                41,
                93,
                116,
                195,
                225,
                25,
                216,
                198,
                74,
                58,
                210,
                17,
                49,
                147,
                92,
                158,
                160,
                75,
                141,
                20,
                96,
                31,
                137,
                117,
                11,
                67,
                214,
                88,
                91,
                24,
            ]

            TEST_LABELS = [
                97,
                218,
                227,
                86,
                217,
                39,
                202,
                87,
                221,
                178,
                40,
                194,
                1,
                71,
                150,
                114,
                56,
                107,
                224,
                179,
                166,
                183,
                50,
                143,
                234,
                154,
                129,
                59,
                55,
                23,
                7,
                8,
                108,
                151,
                22,
                139,
                233,
                173,
                26,
                188,
                35,
                57,
                62,
                70,
                189,
                6,
                28,
                163,
            ]
            class_split_lst = [
                TRAIN_LABELS,
                VAL_LABELS,
                TEST_LABELS,
                TRAIN_LABELS + VAL_LABELS + TEST_LABELS,
            ]
            fs_edges = []
            fs_edge_types = []
            for classes in class_split_lst:
                fs_mask = torch.tensor(
                    [item in classes for item in edge_labels]
                )
                fs_edges.append(edges[:, fs_mask])
                fs_edge_types.append(edge_labels[fs_mask])

            if "wn" in name:
                dataset = WN18RROFADataset("WN18RR", sentence_encoder=encoder)
                class_split_lst = [
                    [
                        0,
                        2,
                        4,
                        5,
                    ],
                    [
                        1,
                        8,
                        9,
                    ],
                    [
                        3,
                        6,
                        7,
                    ],
                ]
                class_split_ratio = [4, 3, 3]
            else:
                raise ValueError("No dataset for test.")

            text_g = dataset.data
            text_g.x = text_g.x_text_feat
            text_g.prompt_edge_feat = dataset.prompt_edge_feat
            converted_triplet = dataset.get_idx_split()
            text_edges = torch.cat(
                [
                    torch.tensor(converted_triplet["train"][0]).T,
                    torch.tensor(converted_triplet["valid"][0]).T,
                    torch.tensor(converted_triplet["test"][0]).T,
                ],
                dim=-1,
            )
            text_g.edge_index = text_edges
            text_edge_labels = torch.cat(
                [
                    torch.tensor(converted_triplet["train"][1]),
                    torch.tensor(converted_triplet["valid"][1]),
                    torch.tensor(converted_triplet["test"][1]),
                ]
            )
            text_g.y = text_edge_labels
            data_manager = FewShotDataManager(
                text_g,
                n_way,
                k_shot,
                q_query,
                class_split_ratio=class_split_ratio,
                class_split_lst=class_split_lst,
            )

            def make_data(
                g,
                data_manager,
                split_name: int,
                state_name,
                undirected_flag,
                edges,
                fs_edges,
                fs_edge_types,
                class_emb_flag=True,
                train_flag=False,
                adj=None,
                random_flag=False,
                min_n=None,
                min_k=None,
            ):
                if class_emb_flag:
                    class_emb = g.edge_label_feat
                else:
                    class_emb = g.prompt_text_feat.repeat(
                        len(g.edge_label_feat), 1
                    )
                if k_shot > 0:
                    data_class = FewShotKGDataset
                elif k_shot == 0:
                    data_class = ZeroShotKGDataset
                else:
                    raise ValueError("check the k_shot value")
                dataset = data_class(
                    pyg_graph=g,
                    class_emb=class_emb,
                    data_idx=torch.zeros(50),
                    n_way=n_way,
                    k_shot=k_shot,
                    q_query=q_query,
                    datamanager=data_manager,
                    mode=split_name,
                    edges=edges,
                    fs_edges=fs_edges,
                    fs_edge_types=fs_edge_types,
                    hop=2,
                    prompt_feat=g.prompt_text_feat,
                    to_undirected=undirected_flag,
                    single_prompt_edge=True,
                    adj=adj,
                    random_flag=random_flag,
                    min_n=min_n,
                    min_k=min_k,
                )
                if train_flag:
                    return dataset
                return DataWithMeta(
                    dataset,
                    batch_size=fs_task_num,
                    sample_size=sample_size,
                    metric="acc",
                    state_name=state_name,
                    classes=min_n,
                    meta_data={"eval_func": classification_single_func},
                )

            if k_shot > 0:
                k_shot_lst = [1, 3, 5]
            else:
                k_shot_lst = [0]

            train_data = [
                make_data(
                    train_g,
                    train_datamanager,
                    0,
                    f"train_fs_fb",
                    True,
                    train_g.edge_index,
                    fs_edges[0],
                    fs_edge_types[0],
                    class_emb_flag=class_emb_flag,
                    train_flag=True,
                    random_flag=True,
                    min_n=min_n,
                    min_k=min_k,
                )
            ]
            valid_data = [
                make_data(
                    g[0],
                    g[1],
                    g[2],
                    f"val_fs{n}{k}_{g[3]}",
                    True,
                    g[4],
                    g[5],
                    g[6],
                    class_emb_flag=class_emb_flag,
                    min_n=n,
                    min_k=k,
                )
                for g in [
                    [
                        train_g,
                        train_datamanager,
                        1,
                        "fb",
                        train_g.edge_index,
                        fs_edges[1],
                        fs_edge_types[1],
                    ],
                    [
                        text_g,
                        data_manager,
                        3,
                        "wn",
                        text_g.edge_index,
                        text_g.edge_index,
                        text_g.y,
                    ],
                ]
                for n in [5, 3, 10]
                for k in k_shot_lst
            ]
            test_data = [
                make_data(
                    g[0],
                    g[1],
                    g[2],
                    f"test_fs{n}{k}_{g[3]}",
                    True,
                    g[4],
                    g[5],
                    g[6],
                    class_emb_flag=class_emb_flag,
                    min_n=n,
                    min_k=k,
                )
                for g in [
                    [
                        train_g,
                        train_datamanager,
                        2,
                        "fb",
                        train_g.edge_index,
                        fs_edges[2],
                        fs_edge_types[2],
                    ],
                    [
                        text_g,
                        data_manager,
                        3,
                        "wn",
                        text_g.edge_index,
                        text_g.edge_index,
                        text_g.y,
                    ],
                ]
                for n in [5, 3, 10]
                for k in k_shot_lst
            ]

            split_data = {
                "train": train_data,
                "val": valid_data,
                "test": test_data,
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

    # fs_setting = {
    #     "fs_arxiv_5_3": [5, 3, 1, 1, False],
    #     # "fs_arxiv_5_1": [5, 1, 1, 1, False],
    #     # "fs_arxiv_5_5": [5, 5, 1, 1, False],
    #     # "fs_arxiv_3_3": [3, 3, 1, 1, False],
    #     # "fs_arxiv_3_1": [3, 1, 1, 1, False],
    #     # "fs_arxiv_3_5": [3, 5, 1, 1, False],
    #     # "fs_arxiv_3_0": [3, 0, 1, 1, True],
    #     "fs_arxiv_5_0": [5, 0, 1, 1, True],
    #     # "fskg_fb_wn_5_3": [5, 3, 1, 1, False],
    #     # "fskg_fb_wn_5_1": [5, 1, 1, 1, False],
    #     # "fskg_fb_wn_5_5": [5, 5, 1, 1, False],
    #     # "fskg_fb_wn_10_3": [10, 3, 1, 1, False],
    #     # "fskg_fb_wn_10_1": [10, 1, 1, 1, False],
    #     # "fskg_fb_wn_10_5": [10, 5, 1, 1, False],
    #     # "fskg_fb_wn_5_0": [5, 0, 1, 1, True],
    #     # "fskg_fb_wn_10_0": [10, 0, 1, 1, True],
    # }
    fs_setting = {
        "fs_arxiv_1": [[5, 3], [5, 1], 1, 1, False],
        "fs_arxiv_0": [[5, 3], [0, 0], 1, 1, True],
        "fskg_fb_wn_1": [[10, 3], [5, 1], 1, 1, False],
        "fskg_fb_wn_0": [[10, 3], [0, 0], 1, 1, True],
    }
    dataset_list = list(fs_setting.keys())
    fs_ds_multiple = [50, 50, 20, 20]
    min_ratio = [50, 0, 50, 0, 20.0, 20.0]

    e2e_data_list = [
        "molzero",
        "molfew",
    ]
    mol_ds_multiple = [0.05] * 2
    mol_ds_min_ratio = [0.05] * 2

    collate_dataset = data_construct(
        dataset_list + e2e_data_list,
        encoder,
        batch_size=params.eval_batch_size,
        sample_size=params.eval_sample_size,
        walk_length=params.rwpe,
        fs_setting=fs_setting,
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
        dataset_multiple=fs_ds_multiple + mol_ds_multiple,
        # dataset_multiple=1,
        patience=3,
        window_size=5,
        min_ratio=min_ratio + mol_ds_min_ratio,
        # min_ratio=1.0,
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
        save_model=True,
        load_best=False,
        reload_freq=1,
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
    params.log_project = "fs_cdm"
    main(params)
