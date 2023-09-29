import os.path as osp
import torch_geometric as pyg
import torch
import json
from data.ofa_data import OFAPygDataset

entity_lst = []
text_lst = []
with open(osp.join(osp.dirname(__file__), "entity2wikidata.json"), "r") as f:
    data = json.load(f)

for k in data:
    # print(data[k])
    entity_lst.append(k)
    text_lst.append(
        "entity nammes: "
        + data[k]["label"]
        + ", entity alternatives: "
        + ", ".join(data[k]["alternatives"])
        + ". entity descriptions:"
        + data[k]["description"]
        if data[k]["description"] is not None
        else "None"
    )


entity2id = {entity: i for i, entity in enumerate(entity_lst)}


def read_knowledge_graph(files):
    relation2id = {}

    converted_triplets = {}
    rel_list = []
    rel = len(relation2id)

    for file_type, file_path in files.items():

        edges = []
        edge_types = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split("\n")[:-1]]
        unknown_entity = 0
        for triplet in file_data:
            if triplet[0] not in entity2id:
                text_lst.append("entity names: Unknown")
                entity_lst.append(triplet[0])
                entity2id[triplet[0]] = len(entity2id)
                unknown_entity += 1
            if triplet[2] not in entity2id:
                text_lst.append("entity names: Unknown")
                entity_lst.append(triplet[2])
                entity2id[triplet[2]] = len(entity2id)
                unknown_entity += 1
            if triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel_list.append(triplet[1])
                rel += 1
            edges.append(
                [
                    entity2id[triplet[0]],
                    entity2id[triplet[2]],
                ]
            )
            edge_types.append(relation2id[triplet[1]])
        print(unknown_entity)
        converted_triplets[file_type] = [edges, edge_types]

    new_data = pyg.data.data.Data(
        x=torch.zeros([len(text_lst), 1]),
        edge_index=torch.tensor(converted_triplets["train"][0]).T,
        edge_types=torch.tensor(converted_triplets["train"][1]),
    )

    node_text = [
        "feature node. entity and entity description: " + ent
        for ent in text_lst
    ]
    edge_text = [
        "feature edge. relation between two entities. " + relation
        for relation in rel_list
    ] + [
        "feature edge. relation between two entities. the inverse relation of "
        + relation
        for relation in rel_list
    ]

    prompt_edge_text = ["prompt edge"]
    prompt_node_text = [
        "prompt node. relation type prediction between the connected entities."
    ]
    label_text = [
        "prompt node. relation between two entities. " + relation
        for relation in rel_list
    ]

    return (
        [new_data],
        [
            node_text,
            edge_text,
            label_text,
            prompt_edge_text,
            prompt_node_text,
        ],
        [converted_triplets, rel_list],
    )


class FB15K237OFADataset(OFAPygDataset):
    def gen_data(self):
        cur_path = osp.dirname(__file__)
        names = ["train", "valid", "test"]
        name_dict = {n: osp.join(cur_path, n + ".txt") for n in names}
        return read_knowledge_graph(name_dict)

    def add_text_emb(self, data_list, text_emb):
        data_list[0].x_text_feat = text_emb[0]
        data_list[0].edge_text_feat = text_emb[1]
        data_list[0].edge_label_feat = text_emb[2]
        data_list[0].prompt_edge_feat = text_emb[3]
        data_list[0].prompt_text_feat = text_emb[4]
        return self.collate(data_list)

    def get_idx_split(self):
        return self.side_data[0]
