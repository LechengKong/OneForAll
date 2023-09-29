import json
import os.path
from abc import abstractmethod
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


class DataManager:
    @abstractmethod
    def get_data_loader(self, mode):
        pass


class FewShotDataManager(DataManager):
    """
    Return dataloader for train/val/test node idx.
    Example:
        data_manager = FewShotDataManager(g, params)
        train_dataloader = data_manager.get_dataloader(0)
        val_dataloader = data_manager.get_dataloader(1)
        test_dataloader = data_manager.get_dataloader(2)
        next(iter(train_dataloader)).shape: (n_way, k_shot + q_query)
    """

    def __init__(
        self,
        data,
        n_way,
        k_shot,
        q_query,
        class_split_ratio=None,
        class_split_lst=None,
        num_workers=0,
    ):
        super(FewShotDataManager, self).__init__()
        data.y = data.y.squeeze()
        self.n_way = n_way
        self.num_workers = num_workers
        self.dataset = FewShotDataset(
            data,
            k_shot + q_query,
            class_split_ratio,
            num_workers=self.num_workers,
            class_split_lst=class_split_lst,
        )
        # self.split = self.dataset.split

    def get_data_loader(self, mode):
        # mode: 0->train, 1->val, 2->test
        class_list = self.dataset.__getclass__(mode)
        sampler = EpisodeBatchSampler(self.n_way, class_list)
        data_loader_params = dict(
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        data_loader = DataLoader(self.dataset, **data_loader_params)
        return data_loader

    def get_dataset(self):
        return self.dataset


class FewShotDataset(Dataset):
    def __init__(
        self, data, batch_size, class_split_ratio, num_workers, class_split_lst
    ):
        self.data = data
        self.batch_size = batch_size
        self.class_split_ratio = class_split_ratio
        self.num_workers = num_workers
        self.class_split_lst = class_split_lst

        self.unique_label = torch.unique(self.data.y)

        self.cls_split_lst = self.class_split()
        self.cls_dataloader = self.create_subdataloader()
        # self.split = self.get_split_index()

    def class_split(self):
        """
        Split class for train/val/test in meta learning setting.
        Save as list: [[train_class_index], [val_class_index], [test_class_index], [all_class_index]]
        """
        # TODO: change data_path
        # cls_split_file = self.args.data_path + '/' + self.args.dataset + '_class_split.json'
        # cls_split_file = os.path.join(os.path.dirname(__file__), 'cache_data', )
        #
        # if os.path.isfile(cls_split_file) and False:
        #     # load list if exists
        #     with open(cls_split_file, 'rb') as f:
        #         cls_split_lst = json.load(f)
        #         print('Complete: Load class split info from %s .' % cls_split_file)
        if self.class_split_lst is not None:
            cls_split_lst = self.class_split_lst + [
                [cls for sublist in self.class_split_lst for cls in sublist]
            ]

        elif self.class_split_ratio is not None:
            # create list according to class_split_ratio and save
            label = self.unique_label.cpu().detach()
            valid_label_mask = torch.where(label >= 0)
            label = label[valid_label_mask]
            selected_labels = []
            if hasattr(self.data, "non_cs_labels"):
                print("Ignore overlapped labels in mag240m with ogbn-arxiv.")
                label = torch.tensor(self.data.non_cs_labels)
                selected_labels = self.data.cs_labels
            # if CoraFull dataset, ignore 68,69 label since they only have 15/29 samples
            # if label.size(0) == 70:
            #     label = label[:-2]
            # randomly shuffle
            label = label.index_select(0, torch.randperm(label.shape[0]))
            train_class, val_class, test_class = torch.split(
                label, self.class_split_ratio
            )
            cls_split_lst = [
                train_class.tolist(),
                val_class.tolist(),
                test_class.tolist(),
                label.tolist(),
                selected_labels,
            ]

        label = self.unique_label.cpu().detach()
        valid_label_mask = torch.where(label >= 0)
        label = label[valid_label_mask]
        # Just for unify with Prodigy (ogbn-arxiv setting)
        if label.shape[0] == 40:
            # TRAIN_LABELS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 15, 17, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33,
            #                 39]
            # VAL_LABELS = [6, 12, 16, 19, 30, 35, 38]
            # TEST_LABELS = [10, 11, 14, 21, 34, 36, 37]

            TRAIN_LABELS = [
                6,
                21,
                22,
                3,
                36,
                11,
                28,
                2,
                7,
                19,
                34,
                29,
                9,
                8,
                25,
                31,
                1,
                23,
                38,
                32,
            ]
            VAL_LABELS = [35, 24, 0, 27, 12, 13, 26, 20, 14, 4]
            TEST_LABELS = [17, 39, 10, 5, 16, 15, 18, 37, 30, 33]
            cls_split_lst = [
                TRAIN_LABELS,
                VAL_LABELS,
                TEST_LABELS,
                TRAIN_LABELS + VAL_LABELS + TEST_LABELS,
            ]

        if label.shape[0] == 237:
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
            cls_split_lst = [
                TRAIN_LABELS,
                VAL_LABELS,
                TEST_LABELS,
                TRAIN_LABELS + VAL_LABELS + TEST_LABELS,
            ]

            # with open(cls_split_file, 'w') as f:
            #     json.dump(cls_split_lst, f)
            #     print('Complete: Save class split info to %s .' % cls_split_file)
        # if len(cls_split_lst[1]) > 0:
        # print(cls_split_lst[:])
        return cls_split_lst

    def label_to_index(self) -> (dict, torch.tensor):
        """
        Generate a dictionary mapping labels to index list
        :return: dictionary: {label: [list of index]}
        """
        label = self.unique_label
        label2index = {}
        remove_label_list = []
        for i in label:
            idx = torch.nonzero(self.data.y == i)
            if idx.shape[0] < self.batch_size * 2:
                remove_label_list.append(int(i))
                label2index[int(i)] = None
            else:
                label2index[int(i)] = idx.squeeze()
        if len(remove_label_list) > 0:
            print(f"Remove invalid labels {len(remove_label_list)}.")
        self.invalid_labels = set(remove_label_list)

        return label2index, label

    def create_subdataloader(self):
        """
        :return: list of subdataloaders for each class i
        """
        label2index, label = self.label_to_index()
        cls_dataloader = []
        cls_dataloader_params = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        for c in label:
            idx = label2index[int(c)]
            if idx is None:
                cls_dataloader.append(False)
            else:
                cls_dataset = ClassDataset(label2index[int(c)])
                cls_dataloader.append(
                    DataLoader(cls_dataset, **cls_dataloader_params)
                )

        return cls_dataloader

    def get_split_index(self):
        """
        :return: dictionary that contains the node index for each split
        """
        label2index, label = self.label_to_index()
        cls_split_lst = self.cls_split_lst
        split = {"train": [], "valid": [], "test": []}

        exclude_labels = []
        for c in label:
            if c in cls_split_lst[0]:
                split["train"].extend(
                    [int(idx) for idx in label2index[int(c)]]
                )
            elif c in cls_split_lst[1]:
                split["valid"].extend(
                    [int(idx) for idx in label2index[int(c)]]
                )
            elif c in cls_split_lst[2]:
                split["test"].extend([int(idx) for idx in label2index[int(c)]])
            else:
                exclude_labels.append(c)

        # print("Ignore labels: " % exclude_labels)

        return split

    def __getitem__(self, class_index):
        node_id = next(iter(self.cls_dataloader[class_index]))
        return next(iter(self.cls_dataloader[class_index])), class_index

    def __len__(self):
        # mode = 0 -> train; 1 -> validation; 2 -> test
        return len(self.unique_label)

    def __getclass__(self, mode):
        # return available classes under current mode (train/val/test)
        # print(self.cls_split_lst)
        class_list = self.cls_split_lst[mode]
        valid_labels = [
            label for label in class_list if label not in self.invalid_labels
        ]
        return valid_labels


class EpisodeBatchSampler(object):
    def __init__(self, n_way, class_list):
        # TODO: change value of episode to some variables
        self.episode = 1
        self.n_way = n_way
        self.class_list = class_list

    def __len__(self):
        return self.episode

    def __iter__(self):
        for i in range(self.episode):
            batch_class = []
            # Don't change task_num for OFA
            task_num = 1
            for j in range(task_num):
                batch_class.append(
                    np.random.choice(
                        self.class_list, self.n_way, replace=False
                    )
                )
            yield np.concatenate(batch_class)


class ClassDataset(Dataset):
    def __init__(self, label_index):
        self.label_index = label_index

    def __getitem__(self, i):
        return self.label_index[i]

    def __len__(self):
        return self.label_index.shape[0]
