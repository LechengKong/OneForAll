from typing import Union, List, Any, Optional, Dict
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from gp.utils.datasets import DatasetWithCollate
from torch_geometric.data import Dataset as PygDataset
from torch_geometric.loader import DataLoader as PygDataloader
from torch.utils.data import Dataset


class DataWithMeta:
    def __init__(
            self,
            data: Dataset,
            batch_size: int,
            state_name: Optional[str] = None,
            feat_dim: int = 0,
            metric: Optional[str] = None,
            classes: Union[int, List[int]] = 2,
            is_regression: bool = False,
            meta_data: Any = None,
            sample_size: Optional[int] = -1,
    ):
        self.data = data
        self.batch_size = batch_size
        self.state_name = state_name
        self.feat_dim = feat_dim
        self.meta_data = meta_data
        self.metric = metric
        self.sample_size = sample_size
        self.classes = classes
        if isinstance(classes, list):
            self.num_tasks = len(classes)
        else:
            self.num_tasks = None
        self.is_regression = is_regression

    def pred_dim(self):
        if self.is_regression:
            return 1
        if self.num_tasks is not None:
            return self.num_tasks
        return self.classes


class DataModule(LightningDataModule):
    def __init__(
            self,
            data: Dict[str, DataWithMeta],
            num_workers: int = 4,
            pin_memory=True,
    ):
        super().__init__()
        self.datasets = data
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def create_dataloader(
            self,
            data: Dataset,
            size: int,
            batch_size: int,
            drop_last: bool = True,
            shuffle: bool = True,
            num_workers: int = 0,
    ):
        sampler = None
        if size > 0:
            sampler = RandomSampler(data, num_samples=size, replacement=True)
            shuffle = False
        if isinstance(data, DatasetWithCollate):
            return DataLoader(
                data,
                batch_size,
                shuffle,
                sampler,
                num_workers=num_workers,
                collate_fn=data.get_collate_fn(),
                drop_last=drop_last,
                pin_memory=self.pin_memory,
            )
        if isinstance(data, PygDataset):
            return PygDataloader(
                data,
                batch_size,
                shuffle,
                sampler=sampler,
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=self.pin_memory,
            )

    def train_dataloader(self):
        return self.create_dataloader(
            self.datasets["train"].data,
            self.datasets["train"].sample_size,
            self.datasets["train"].batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if isinstance(self.datasets["val"], list):
            data_list = []
            for val_data in self.datasets["val"]:
                data_list.append(
                    self.create_dataloader(
                        val_data.data,
                        val_data.sample_size,
                        val_data.batch_size,
                        drop_last=False,
                        shuffle=False,
                        num_workers=self.num_workers,
                    )
                )
            return data_list
        else:
            return [
                self.create_dataloader(
                    self.datasets["val"].data,
                    self.datasets["val"].sample_size,
                    self.datasets["val"].batch_size,
                    drop_last=False,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
            ]

    def test_dataloader(self):
        if isinstance(self.datasets["test"], list):
            data_list = []
            for test_data in self.datasets["test"]:
                data_list.append(
                    self.create_dataloader(
                        test_data.data,
                        test_data.sample_size,
                        test_data.batch_size,
                        drop_last=False,
                        shuffle=False,
                        num_workers=self.num_workers,
                    )
                )
            return data_list
        else:
            return [
                self.create_dataloader(
                    self.datasets["test"].data,
                    self.datasets["test"].sample_size,
                    self.datasets["test"].batch_size,
                    drop_last=False,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
            ]
