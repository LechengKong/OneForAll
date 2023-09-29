import numpy as np
from scipy.sparse import csr_matrix
import os.path as osp
import os
import torch
import pickle as pkl
import yaml
from tqdm import tqdm

from gp.utils.graph import construct_dgl_graph_from_edges


def read_knowledge_graph(files, relation2id=None):
    entity2id = {}
    if relation2id is None:
        relation2id = {}

    converted_triplets = {}
    rel_list = [[] for i in range(len(relation2id))]

    ent = 0
    rel = len(relation2id)

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split("\n")[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1
                rel_list.append([])

            data.append(
                [
                    entity2id[triplet[0]],
                    relation2id[triplet[1]],
                    entity2id[triplet[2]],
                ]
            )

        for trip in data:
            rel_list[trip[1]].append([trip[0], trip[2]])

        converted_triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    adj_list = []
    for rel_mat in rel_list:
        rel_array = np.array(rel_mat)
        if len(rel_array) == 0:
            adj_list.append(csr_matrix((len(entity2id), len(entity2id))))
        else:
            adj_list.append(
                csr_matrix(
                    (
                        np.ones(len(rel_mat)),
                        (rel_array[:, 0], rel_array[:, 1]),
                    ),
                    shape=(len(entity2id), len(entity2id)),
                )
            )

    return (
        adj_list,
        converted_triplets,
        entity2id,
        relation2id,
        id2entity,
        id2relation,
    )


def save_load_torch_data(
    folder_path,
    data,
    num_output=1,
    data_fold=5,
    data_name="saved_gd_data",
    num_workers=32,
):
    saved_data_path = osp.join(folder_path, data_name)
    if not osp.exists(saved_data_path):
        os.mkdir(saved_data_path)
        dt = torch.utils.data.DataLoader(
            data,
            batch_size=256,
            num_workers=32,
        )
        pbar = tqdm(dt)
        fold_len = int(len(dt) / (data_fold - 1))
        count = 0
        fold_count = 0
        for i, t in enumerate(pbar):
            if count == 0:
                data_col = []
                for j in range(num_output):
                    data_col.append([])
            if num_output == 1:
                data_col[0].append(t)
            else:
                for j, v in enumerate(t):
                    data_col[j].append(v)
            count += 1
            if count == fold_len:
                for i, it in enumerate(data_col):
                    cdata = torch.cat(it, dim=0).numpy()
                    np.save(
                        osp.join(
                            saved_data_path, str(i) + "_" + str(fold_count)
                        ),
                        cdata,
                    )
                    for itm in it:
                        del itm
                    del cdata
                fold_count += 1
                count = 0
        if count > 0:
            for i, it in enumerate(data_col):
                cdata = torch.cat(it, dim=0).numpy()
                np.save(
                    osp.join(saved_data_path, str(i) + "_" + str(fold_count)),
                    cdata,
                )
    saved_data = []
    if num_output == 1:
        for j in range(data_fold):
            ipath = osp.join(saved_data_path, str(0) + "_" + str(j) + ".npy")
            if osp.exists(ipath):
                saved_data.append(np.load(ipath))
    else:
        for i in range(num_output):
            saved_data.append([])
            for j in range(data_fold):
                ipath = osp.join(
                    saved_data_path, str(i) + "_" + str(j) + ".npy"
                )
                if osp.exists(ipath):
                    saved_data[i].append(np.load(ipath))
    return saved_data, data_fold


def load_exp_dataset_dgl(directory):
    graphs = []
    labels = []
    with open(directory, "r") as data:
        num_graphs = int(data.readline().rstrip().split(" ")[0])
        for i in range(num_graphs):
            graph_meta = data.readline().rstrip().split(" ")
            num_vertex = int(graph_meta[0])
            # curr_graph = np.zeros(shape=(num_vertex, num_vertex))
            labels.append(int(graph_meta[1]))
            node_labels = np.zeros((num_vertex, 2))
            edges = []
            for j in range(num_vertex):
                vertex = data.readline().rstrip().split(" ")
                node_labels[j, int(vertex[0])] = 1
                for k in range(2, len(vertex)):
                    edges.append([j, int(vertex[k])])
            edges = np.array(edges)
            g = construct_dgl_graph_from_edges(
                edges[:, 0],
                edges[:, 1],
                n_entities=num_vertex,
                inverse_edge=True,
            )
            g.ndata["feat"] = torch.tensor(node_labels, dtype=torch.float)
            graphs.append(g)
    return graphs, np.array(labels)


def open_and_load_pickle(filename):
    open_file = open(filename, "rb")
    data = pkl.load(open_file)
    open_file.close()
    return data


def load_yaml(dir):
    with open(dir, "r") as stream:
        return yaml.safe_load(stream)
