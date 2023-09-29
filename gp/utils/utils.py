from itertools import product
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import random
import os
import json
import yaml

import os.path as osp
from datetime import datetime
from gp.utils.io import load_yaml
from types import SimpleNamespace


class SmartTimer:
    """A timer utility that output the elapsed time between this
    call and last call.
    """

    def __init__(self, verb=True) -> None:
        """SmartTimer Constructor

        Keyword Arguments:
            verb {bool} -- Controls printing of the timer (default: {True})
        """
        self.last = time.time()
        self.verb = verb

    def record(self):
        """Record current timestamp"""
        self.last = time.time()

    def cal_and_update(self, name):
        """Record current timestamp and print out time elapsed from last
        recorded time.

        Arguments:
            name {string} -- identifier of the printout.
        """
        now = time.time()
        if self.verb:
            print(name, now - self.last)
        self.record()


class SparseData:
    def __init__(self, data, data_count=None, data_offset=None):
        self.data = data
        if data_count is None and data_offset is None:
            if isinstance(data, list):
                self.num_data = len(data)
                self.data_count = np.array([len(d) for d in data])
                self.data = np.concatenate(data, axis=0)
            elif isinstance(data, np.ndarray):
                self.num_data = 1
                self.data_count = len(data)
            self.data_offset = self.count2offset(self.data_count)
        if self.data_count is None:
            self.data_count = self.offset2count(self.data_offset)
        if self.data_offset is None:
            self.data_offset = self.count2offset(self.data_count)

    def count2offset(self, count):
        return np.r_[0, np.cumsum(count[:-1])]

    def offset2count(self, offset):
        return np.r_[offset[1:], len(self.data)] - offset


def save_params(filename, params):
    """Write a Namespace object to file

    Arguments:
        filename {string} -- destination of the saved file
        params {Namespace} -- namespace object
    """
    d = vars(params)
    with open(filename, "a") as f:
        json.dump(d, f)


def set_random_seed(seed):
    """Set python, numpy, pytorch global random seed.
    Does not guarantee determinism due to PyTorch's feature.

    Arguments:
        seed {int} -- Random seed to set
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sparse_uniform_select(data, data_count, p=0.5):
    """Randomly select from a sparse representation.

    Arguments:
        data {numpy.ndarray} -- Sparse data
        data_count {numpy.ndarray} -- Sparse index of the data

    Keyword Arguments:
        p {int or numpy.ndarray} -- If int, mu=p for all entries,
        if numpy.ndarray, should be of the same shape as data_count.
         (default: {0.5})

    Returns:
        selected_data -- selected_data
    """
    if isinstance(p, np.ndarray):
        p = np.repeat(p, data_count)
    prob = np.random.rand(len(data))
    select = prob < p
    data_index = np.arange(len(data_count)).repeat(data_count)
    new_data_count = np.bincount(data_index[select], minlength=len(data_count))
    return data[select], new_data_count


def sparse_uniform_sample(data, data_count, c=1):
    """
    Return a DGL graph specified by edges.

    :param head: Edge head.
    :param tail: Edge tail.
    :return: DGLGraph.
    :rtype: DGLGraph
    """
    if isinstance(c, int):
        c = np.repeat(c, len(data_count))
        c = (data_count > 0) * c
    max_val = np.max(data_count) * 10
    select_ind = np.random.randint(max_val, size=np.sum(c))
    select_ind = select_ind % np.repeat(data_count, c)
    offset = np.r_[0, data_count[:-1]]
    offset = np.cumsum(offset)
    select_ind = select_ind + offset.repeat(c)
    return data[select_ind], c


def k_fold_ind(labels, fold):
    """Generate stratified k fold split index based on labels

    Arguments:
        labels {np.ndarray} -- labels of the data
        fold {int} -- number of folds

    Returns:
        list[numpy.ndarray] -- A list whose elements are indices of data
        in the fold.
    """
    ksfold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=10)
    folds = []
    for _, t_index in ksfold.split(
        np.zeros_like(np.array(labels)), np.array(labels, dtype=int)
    ):
        folds.append(t_index)
    return folds


def k_fold2_split(folds, data_len):
    """Split the data index into train/test/validation based on fold,
    one fold for testing, one fold for validation and the rest for training.

    Arguments:
        folds {list[numpy.ndarray]} -- fold information
        data_len {int} -- lenght of the data

    Returns:
        list[list[numpy.ndarray]] -- a list of train/test/validation split
        indices.
    """
    splits = []
    for i in range(len(folds)):
        test_arr = np.zeros(data_len, dtype=bool)
        test_arr[folds[i]] = 1
        val_arr = np.zeros(data_len, dtype=bool)
        val_arr[folds[int((i + 1) % len(folds))]] = 1
        train_arr = np.logical_not(np.logical_or(test_arr, val_arr))
        train_ind = train_arr.nonzero()[0]
        test_ind = test_arr.nonzero()[0]
        val_ind = val_arr.nonzero()[0]
        splits.append([train_ind, test_ind, val_ind])
    return splits


def dict_res_summary(res_col):
    """Combine multiple dictionary information into one dictionary
    so that all entries with the same key will be concatenated into
    a list

    Arguments:
        res_col {list[dictionary]} -- a list of dictionary

    Returns:
        dictionary -- summarized dictionary information
    """
    res_dict = {}
    for res in res_col:
        for k in res:
            if k not in res_dict:
                res_dict[k] = []
            res_dict[k].append(res[k])
    return res_dict


def multi_data_average_exp(data, args, exp):
    val_res_col = []
    test_res_col = []
    for split in data:
        val_res, test_res = exp(split, args)
        val_res_col.append(val_res)
        test_res_col.append(test_res)

    val_res_dict = dict_res_summary(val_res_col)
    test_res_dict = dict_res_summary(test_res_col)
    return val_res_dict, test_res_dict


def hyperparameter_grid_search(
    hparams, data, exp, args, search_metric, evaluator, exp_arg=None
):
    named_params = [[(k, p) for p in hparams[k]] for k in hparams]
    best_met = evaluator.init_result()
    best_res = None
    params = product(*named_params)
    for p in params:
        for name, val in p:
            setattr(args, name, val)
        if exp_arg:
            val_res, test_res = exp(data, args, exp_arg)
        else:
            val_res, test_res = exp(data, args)
        val_metric_res, test_metric_res = np.array(
            val_res[search_metric]
        ), np.array(test_res[search_metric])
        val_mean, val_std = np.mean(val_metric_res), np.std(val_metric_res)
        test_mean, test_std = np.mean(test_metric_res), np.std(test_metric_res)
        if evaluator.better_results(val_mean, best_met):
            best_met = val_mean
            best_res = {
                "metric": search_metric,
                "val_mean": val_mean,
                "val_std": val_std,
                "test_mean": test_mean,
                "test_std": test_std,
                "full_val": val_res,
                "full_test": test_res,
                "params": p,
            }
    return best_res


def write_res_to_file(
    file,
    dataset,
    metric,
    test_mean,
    val_mean=0,
    test_std=0,
    val_std=0,
    params=None,
    res=None,
):
    with open(file, "a") as f:
        f.write("\n\n")
        f.write(res)
        f.write("\n")
        f.write("Dataset: {} \n".format(dataset))
        f.write("Optimize wrt {}\n".format(metric))
        f.write("val, {:.5f} ± {:.5f} \n".format(val_mean, val_std))
        f.write("test, {:.5f} ± {:.5f} \n".format(test_mean, test_std))
        f.write("best res:")
        f.write(str(params))


def var_size_repeat(size, chunks, repeats):
    a = np.arange(size)
    s = np.r_[0, chunks.cumsum()]
    starts = a[np.repeat(s[:-1], repeats)]
    if len(starts) == 0:
        return np.array([], dtype=int)
    chunk_rep = np.repeat(chunks, repeats)
    ends = starts + chunk_rep

    clens = chunk_rep.cumsum()
    ids = np.ones(clens[-1], dtype=int)
    ids[0] = starts[0]
    ids[clens[:-1]] = starts[1:] - ends[:-1] + 1
    out = ids.cumsum()
    return out


def count_to_group_index(count):
    return torch.arange(len(count), device=count.device).repeat_interleave(
        count
    )


def setup_exp(params):
    if not osp.exists("./saved_exp"):
        os.mkdir("./saved_exp")

    curtime = datetime.now()
    exp_dir = osp.join("./saved_exp", str(curtime))
    os.mkdir(exp_dir)
    with open(osp.join(exp_dir, "command"), "w") as f:
        yaml.dump(params, f)
    params["exp_dir"] = exp_dir


def combine_dict(*args):
    combined_dict = {}
    for d in args:
        for k in d:
            combined_dict[k] = d[k]
    return combined_dict


def merge_mod(params, mod_args):
    for i in range(0, len(mod_args), 2):
        if mod_args[i + 1].isdigit():
            val = int(mod_args[i + 1])
        elif mod_args[i + 1].replace(".", "", 1).isdigit():
            val = float(mod_args[i + 1])
        elif mod_args[i + 1].lower() == "true":
            val = True
        elif mod_args[i + 1].lower() == "false":
            val = False
        else:
            val = mod_args[i + 1]
        params[mod_args[i]] = val
    return params


def convert_yaml_params(params_path):
    load_params = load_yaml(params_path)
    load_params = SimpleNamespace(**load_params)
    return load_params
