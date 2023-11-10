import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaModel,
    LlamaConfig,
)
from torchmetrics import AveragePrecision, AUROC
import numpy as np
from torch_geometric.utils import (
    to_scipy_sparse_matrix,
    scatter,
)


class SentenceEncoder:
    def __init__(
            self, name, root="cache_data/model", batch_size=512, multi_gpu=False
    ):
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        if name == "ST":
            self.model = SentenceTransformer(
                "multi-qa-distilbert-cos-v1",
                device="cuda:0",
                cache_folder=root,
            )
        elif name == "llama2":
            model_path = os.path.join(os.getcwd(), "llama-2-7b")
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_path, device="cuda:0"
            )
            self.model = LlamaModel.from_pretrained(model_path).to("cuda:0")
        elif name == "roberta":
            self.model = SentenceTransformer(
                "sentence-transformers/roberta-base-nli-stsb-mean-tokens",
                device="cuda:0",
                cache_folder=root,
            )
        else:
            raise ValueError(f"Unknown language model: {name}.")

    def encode(self, texts, to_tensor=True):
        if self.multi_gpu:
            # Start the multi-process pool on all available CUDA devices
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(
                texts,
                pool=pool,
                batch_size=self.batch_size,
            )
            embeddings = torch.from_numpy(embeddings)
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_tensor=to_tensor,
                convert_to_numpy=not to_tensor,
            )
        return embeddings


def binary_single_auc_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    score = torch.sigmoid(output)
    # if len(score.unique()) == 1:
    # print(output[:20])
    label = batch.bin_labels[batch.true_nodes_mask]
    # print(score)
    # print(label)
    return func.update(score, label.view(-1, batch.num_classes[0]))


def binary_apr_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    score = torch.sigmoid(output)
    label = batch.bin_labels[batch.true_nodes_mask]
    return func.update(score, label.view(len(batch), -1))


def binary_auc_multi_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    score = torch.sigmoid(output)
    label = batch.bin_labels[batch.true_nodes_mask]
    return func.update(score, label.view(-1, batch.num_classes[0]))


def label_apr_func(func, output, batch):
    score = torch.sigmoid(output)
    return func.update(score, batch.y)


def flat_label_func(func, output, batch):
    labels = batch.y.view(-1)
    valid_ind = labels == labels
    return func(output.view(-1)[valid_ind], labels[valid_ind])


def classification_single_func(func, output, batch):
    label = batch.bin_labels[batch.true_nodes_mask].view(-1, batch.num_classes[0])
    output = output.view(-1, batch.num_classes[0])
    return func(output, torch.argmax(label, dim=-1))


class MultiApr(torch.nn.Module):
    def __init__(self, num_labels=1):
        super().__init__()
        self.metrics = torch.nn.ModuleList(
            [AveragePrecision(task="binary") for i in range(num_labels)]
        )

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            # print(pred[valid_idx])
            # print(target[valid_idx])
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()


class MultiAuc(torch.nn.Module):
    def __init__(self, num_labels=1):
        super().__init__()
        self.metrics = torch.nn.ModuleList(
            [AUROC(task="binary") for i in range(num_labels)]
        )

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            # print(pred[valid_idx])
            # print(target[valid_idx])
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()


def scipy_rwpe(data, walk_length):
    row, col = data.edge_index
    N = data.num_nodes

    value = data.edge_weight
    if value is None:
        value = torch.ones(data.num_edges, device=row.device)
    value = scatter(value, row, dim_size=N, reduce="sum").clamp(min=1)[row]
    value = 1.0 / value
    adj = to_scipy_sparse_matrix(
        data.edge_index, edge_attr=value, num_nodes=data.num_nodes
    )

    out = adj
    pe_list = [out.diagonal()]
    for _ in range(walk_length - 1):
        out = out @ adj
        pe_list.append(out.diagonal())
    pe = torch.tensor(np.stack(pe_list, axis=-1))

    return pe


def get_label_texts(labels):
    label_texts = [None] * int(len(labels) * 2)
    for entry in labels:
        label_texts[labels[entry][0]] = (
                "prompt node. molecule property description. "
                + "The molecule is effective to the following assay. "
                + labels[entry][1][0][:-41]
        )
        label_texts[labels[entry][0] + len(labels)] = (
                "prompt node. molecule property description. "
                + "The molecule is not effective to the following assay. "
                + labels[entry][1][0][:-41]
        )
    return label_texts
