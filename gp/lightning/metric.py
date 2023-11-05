import torch
import os.path as osp
import copy
from typing import Any, Callable, Optional, Literal, List, Union
from torchmetrics import MeanAbsoluteError, Accuracy, AUROC, MeanMetric, Metric
from torchmetrics.text import BLEUScore
from itertools import chain


def classification_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    return func(output, batch.y.view(-1).to(torch.long))


def similarity_func(func, output, batch):
    return -func(output[0], output[1], dim=-1).mean()


def flat_binary_func(func, output, batch):
    labels = batch.bin_labels[batch.true_nodes_mask]
    valid_ind = labels == labels
    return func(output.view(-1)[valid_ind], labels[valid_ind])


def flat_binary_func_fs(func, output, batch):
    labels = batch.bin_labels
    valid_ind = labels == labels
    return func(output.view(-1)[valid_ind], labels[valid_ind])


def binary_auc_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    # score = torch.sigmoid(output)[:, -1]
    score = torch.nn.functional.softmax(output, dim=-1)[:, -1]
    return func(score, batch.y[:, -1].view(-1))


def regression_func(func, output, batch):
    return func(output.view(-1), batch.y.view(-1))


def batch_contrast(inp1, inp2, temp):
    anchor_emb = inp1.repeat_interleave(len(inp2), dim=0)
    neg_emb = inp2.repeat([len(inp1), 1])
    sim = torch.sum(anchor_emb * neg_emb, dim=-1) / temp
    sim = torch.exp(sim).view(len(inp1), len(inp2))
    print(sim)
    return -torch.log(torch.diagonal(sim) / sim.sum(dim=-1)).mean()


def get_contrast_func(temprature, cl_mode):
    if cl_mode == "twoview":

        def contrast_loss_func(loss, output, batch):
            return batch_contrast(output[0], output[1], temprature)

    elif cl_mode == "self":

        def contrast_loss_func(loss, output, batch):
            return batch_contrast(output, output, temprature)

    return contrast_loss_func


def generation_func(loss, output, batch):
    y = batch.y.to(torch.long).view(-1)
    return loss(output, y)


def BLEU_func(metric, output, batch):
    combine_texts = batch.combine_texts
    combine_texts = list(chain.from_iterable(combine_texts))
    combine_texts = [[text] for text in combine_texts]
    output_texts = batch.output_texts
    output_texts = list(chain.from_iterable(output_texts))
    output_texts = [[text] for text in output_texts]
    print(combine_texts[0])
    print(output[0])
    print("=" * 89)
    print(combine_texts[1])
    print(output[1])
    print("=" * 89)
    print(combine_texts[2])
    print(output[2])
    print("=" * 89)

    return metric(output, output_texts)


class EvalKit(torch.nn.Module):
    def __init__(
        self,
        metric_name: Union[str, List[str]],
        evlter: Any,
        loss: Any,
        evlter_func: Union[Callable, List[Callable]] = None,
        loss_func: Callable = None,
        val_monitor_state: Optional[str] = "valid",
        test_monitor_state: Optional[str] = "test",
        eval_mode: Literal["min", "max"] = "min",
        exp_prefix: str = "",
        eval_state: List[str] = ["train_eval", "test", "valid"],
    ):
        super().__init__()
        self.eval_states = eval_state
        self.loss = loss
        self.eval_mode = eval_mode
        self.val_monitor_state = val_monitor_state
        self.test_monitor_state = test_monitor_state
        self.exp_prefix = exp_prefix
        self.evlters = torch.nn.ModuleDict()
        self.loss_func = loss_func
        self.evlter_func = {}
        self.metric_name = {}
        for i, state in enumerate(eval_state):
            if isinstance(evlter, Metric):
                self.metric_name[state] = osp.join(
                    exp_prefix, state, metric_name
                )
                self.evlters[state] = copy.deepcopy(evlter)
                self.evlter_func[state] = evlter_func
            else:
                self.metric_name[state] = osp.join(
                    exp_prefix, state, metric_name[i]
                )
                self.evlters[state] = evlter[i]
                self.evlter_func[state] = evlter_func[i]

        self.val_metric = self.metric_name[self.val_monitor_state]
        self.test_metric = self.metric_name[self.test_monitor_state]

    def compute_loss(self, output: Any, batch: Any):
        return self.loss_func(self.loss, output, batch)

    def has_eval_state(self, state: str):
        return state in self.eval_states

    def get_evlter(self, state: str):
        return self.evlters[state]

    def eval_step(self, output: Any, batch: Any, state: str):
        evlter = self.get_evlter(state)
        return self.evlter_func[state](evlter, output, batch)

    def eval_epoch(self, state: str):
        evlter = self.get_evlter(state)
        return evlter.compute()

    def eval_reset(self, state: str):
        evlter = self.get_evlter(state)
        evlter.reset()

    def get_metric_name(self, state: str):
        return self.metric_name[state]


def prepare_mae(exp_name, eval_state=["train_eval", "test", "valid"]):
    evlter = MeanAbsoluteError()
    loss = torch.nn.L1Loss()
    return EvalKit(
        "mae",
        evlter,
        loss,
        regression_func,
        regression_func,
        exp_prefix=exp_name,
        eval_state=eval_state,
    )


def prepare_auc(exp_name, eval_state=["train_eval", "test", "valid"]):
    evlter = AUROC(task="binary")
    loss = torch.nn.CrossEntropyLoss()
    return EvalKit(
        "auc",
        evlter,
        loss,
        binary_auc_func,
        classification_func,
        eval_mode="max",
        exp_prefix=exp_name,
        eval_state=eval_state,
    )


def prepare_acc(
    exp_name, eval_state=["train_eval", "test", "valid"], **kwargs
):
    loss = torch.nn.CrossEntropyLoss()
    evlter = Accuracy(task="multiclass", num_classes=kwargs["num_class"])

    return EvalKit(
        "acc",
        evlter,
        loss,
        classification_func,
        classification_func,
        eval_mode="max",
        exp_prefix=exp_name,
        eval_state=eval_state,
    )


def prepare_bin_acc(
    exp_name, eval_state=["train_eval", "test", "valid"], **kwargs
):
    loss = torch.nn.BCEWithLogitsLoss()
    evlter = Accuracy(task="multiclass", num_classes=kwargs["num_class"])

    return EvalKit(
        "acc",
        evlter,
        loss,
        classification_func,
        flat_binary_func,
        eval_mode="max",
        exp_prefix=exp_name,
        eval_state=eval_state,
        val_state="valid_0",
    )


def prepare_bin_auc(
    exp_name, eval_state=["train_eval", "test", "valid"], **kwargs
):
    loss = torch.nn.BCEWithLogitsLoss()
    evlter = AUROC(task="binary")

    return EvalKit(
        "auc",
        evlter,
        loss,
        binary_auc_func,
        flat_binary_func,
        eval_mode="max",
        exp_prefix=exp_name,
        eval_state=eval_state,
    )


def prepare_cl(exp_name, eval_state=["train_eval", "test", "valid"], **kwargs):
    evlter = MeanMetric()

    return EvalKit(
        "acc",
        evlter,
        None,
        get_contrast_func(kwargs["temprature"], kwargs["mode"]),
        get_contrast_func(kwargs["temprature"], kwargs["mode"]),
        eval_mode="max",
        exp_prefix=exp_name,
        eval_state=eval_state,
    )


class IdentityLoss(torch.nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()

    def forward(self, inputs, targets):
        return inputs


def prepare_generation(exp_name, eval_state=["test", "valid"], **kwargs):
    loss = IdentityLoss()
    evlter = BLEUScore()
    return EvalKit(
        "BLEU",
        evlter,
        loss,
        BLEU_func,
        generation_func,
        eval_mode="min",
        exp_prefix=exp_name,
        eval_state=eval_state,
    )


available_metrics = {
    "acc": prepare_acc,
    "mae": prepare_mae,
    "auc": prepare_auc,
    "cl": prepare_cl,
    "generation": prepare_generation,
}


def prepare_metric(metric, params, exp_name, data, eval_state=None):
    if metric not in available_metrics:
        raise NotImplementedError(metric + " not a available metric")
    if eval_state is not None:
        return available_metrics[metric](params, exp_name, data, eval_state)
    else:
        return available_metrics[metric](params, exp_name, data)
