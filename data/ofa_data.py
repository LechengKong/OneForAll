import os
import os.path as osp
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Dataset



def safe_mkdir(path):
    if not osp.exists(path):
        os.mkdir(path)


def pth_safe_save(obj, path):
    if obj is not None:
        torch.save(obj, path)


def pth_safe_load(path):
    if osp.exists(path):
        return torch.load(path)
    return None


class OFAPygDataset(InMemoryDataset):
    def __init__(
        self,
        name,
        encoder,
        root="./cache_data",
        load_text=False,
        load_feat=True,
        transform=None,
        pre_transform=None,
        meta_dict=None,
    ):

        self.name = name
        self.root = root
        self.data_dir = osp.join(self.root, self.name)
        self.encoder = encoder
        super().__init__(self.data_dir, transform, pre_transform)
        safe_mkdir(self.data_dir)

        if load_text:
            self.texts = torch.load(self.processed_paths[1])

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.side_data = pth_safe_load(self.processed_paths[2])

    def data2vec(self, data):
        if self.encoder is None:
            raise NotImplementedError("LLM encoder is not defined")
        if data is None:
            return None
        embeddings = self.encoder.encode(data).cpu()
        return embeddings

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["geometric_data_processed.pt", "texts.pkl", "data.pt"]

    def text2feature(self, texts):
        if isinstance(texts[0], str):
            return self.data2vec(texts)
        return [self.text2feature(t) for t in texts]

    def gen_data(self):
        pass

    def add_text_emb(self, data_list, texts_emb):
        pass


    def process(self):
        self.encoder.get_model()
        data_list, texts, side_data = self.gen_data()

        texts_emb = self.text2feature(texts)
        torch.save(texts, self.processed_paths[1])
        if side_data is not None:
            torch.save(side_data, self.processed_paths[2])
        else:
            torch.save("No side data", self.processed_paths[2])
        data, slices = self.add_text_emb(data_list, texts_emb)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])


class OFAPygSTDataset(OFAPygDataset):
    def __init__(
        self,
        name,
        encoder,
        root="./cache_data",
        load_text=False,
        load_feat=True,
        transform=None,
        pre_transform=None,
        meta_dict=None,
    ):

        self.name = name
        self.encoder = encoder
        self.root = root
        self.data_dir = osp.join(self.root, self.name)
        safe_mkdir(self.data_dir)
        super().__init__(self.data_dir, transform, pre_transform)

        if load_text:
            self.texts = torch.load(self.processed_paths[0])

        self.side_data = pth_safe_load(self.processed_paths[1])
        self.global_data = pth_safe_load(self.processed_paths[2])

        self.convert_data()

    @property
    def processed_file_names(self):
        return [
            "texts.pkl",
            "data.pt",
            "global_data.pt",
            "node_feat.npy",
            "edge_feat.npy",
        ]

    def len(self):
        return 0

    def convert_data(self):
        pass

    def process(self):
        self.encoder.get_model()
        data_list, texts, side_data = self.gen_data()
        texts_emb = self.text2feature(texts)
        torch.save(texts, self.processed_paths[0])
        if side_data is not None:
            torch.save(side_data, self.processed_paths[1])
        else:
            torch.save("No side data", self.processed_paths[1])
        data, global_data = self.add_text_emb(data_list, texts_emb)
        if global_data is not None:
            torch.save(global_data, self.processed_paths[2])
        else:
            torch.save("No global data", self.processed_paths[2])

        print("Saving...")

        node_memmap = np.memmap(
            self.processed_paths[3],
            dtype="float32",
            mode="w+",
            shape=data[0].shape,
        )
        node_memmap[:] = data[0]
        node_memmap.flush()

        edge_memmap = np.memmap(
            self.processed_paths[4],
            dtype="float32",
            mode="w+",
            shape=data[1].shape,
        )
        edge_memmap[:] = data[1]
        edge_memmap.flush()

    def get(self, idx):
        data = torch.load(self.processed_paths[idx + 3])
        return data