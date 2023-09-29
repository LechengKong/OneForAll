import torch_geometric as pyg
from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset



class DatasetWithCollate(Dataset, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def get_collate_fn(self):
        pass


class DGLSingleGraphDataset(DatasetWithCollate):
    def __init__(self, graph):
        super().__init__()
        self.num_nodes = graph.num_nodes()
        self.graph = graph
        self.adj_mat = self.graph.adjacency_matrix(
            transpose=False, scipy_fmt="csr"
        )


class PyGSingleGraphDataset(DatasetWithCollate):
    def __init__(self, graph: pyg.data):
        super().__init__()
        self.num_nodes = graph.num_nodes()
        self.graph = graph
        self.adj_mat = pyg.utils.to_scipy_sparse_matrix(
            graph.edge_index
        ).tocsr()