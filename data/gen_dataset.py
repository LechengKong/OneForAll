import json
import os.path
from abc import abstractmethod

# utils
import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader, Dataset
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    CoraFull,
    Coauthor,
    WikiCS,
)
from tqdm.autonotebook import trange

import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected

from sentence_transformers import SentenceTransformer
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaModel,
    LlamaConfig,
)


"""
For Debug:
from data.gen_dataset import load_dataset
    dataset = load_dataset('arxiv','llama2')
"""


def load_dataset(name, lm_name):
    """
    Return PyG dataset
    encoded node feature -> 'xdesc'
    encoded category feature -> 'ydesc'
    """
    dataset_dir = os.path.join(
        os.path.dirname(__file__), "../cache_data", name
    )
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    available_datasets = ["arxiv", "wikics"]

    # Load dataset from PyG
    if name == "arxiv":
        from data.arxiv.gen_data import get_text

        dataset = PygNodePropPredDataset(name="ogbn-" + name, root=dataset_dir)
        # change arxiv to undirected graph
        edge_index = to_undirected(dataset._data.edge_index)
        set_dataset_attr(
            dataset, "edge_index", edge_index, edge_index.shape[1]
        )
    elif name == "wikics":
        from data.wikics.gen_data import get_text

        dataset = WikiCS(root=dataset_dir)
    else:
        raise ValueError(
            f"Unknown dataset: {name}. Please choose from {available_datasets}."
        )

    if not (
        os.path.exists(
            os.path.join(dataset_dir, f"{name}_{lm_name}_node_emb.pth")
        )
        or os.path.exists(
            os.path.join(dataset_dir, f"{name}_{lm_name}_class_emb.pth")
        )
    ):

        # Load raw text
        node_text_lst, class_text_lst = get_text(name)

        # Generate text embeddings
        print("Generating Embeddings.")
        node_embeddings, class_embeddings = text_model(
            node_text_lst, class_text_lst, name=lm_name
        )

        # Save embeddings
        torch.save(
            node_embeddings,
            os.path.join(dataset_dir, f"{name}_{lm_name}_node_emb.pth"),
        )
        torch.save(
            class_embeddings,
            os.path.join(dataset_dir, f"{name}_{lm_name}_class_emb.pth"),
        )

    else:
        print("Loading Embeddings.")
        node_embeddings = torch.load(
            os.path.join(dataset_dir, f"{name}_{lm_name}_node_emb.pth"),
            map_location="cpu",
        )
        class_embeddings = torch.load(
            os.path.join(dataset_dir, f"{name}_{lm_name}_class_emb.pth"),
            map_location="cpu",
        )

    # add node text feature embedding
    set_dataset_attr(
        dataset, "xdesc", node_embeddings, node_embeddings.shape[0]
    )
    # add label text feature embedding
    set_dataset_attr(
        dataset, "ydesc", class_embeddings, class_embeddings.shape[0]
    )

    return dataset


def text_model(node_text_lst, class_text_lst, name="ST"):
    if name == "ST":
        model = SentenceTransformer(
            "multi-qa-distilbert-cos-v1", device="cuda:0", cache_folder=os.path.join(os.path.dirname(__file__), "../cache_data")
        )
        node_embeddings = model.encode(
            node_text_lst,
            batch_size=128,
            show_progress_bar=True,
            convert_to_tensor=True,
        ).cpu()
        class_embeddings = model.encode(
            class_text_lst, show_progress_bar=True, convert_to_tensor=True
        ).cpu()
    elif name == "llama2":
        # model_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'llama-2-7b')
        model_path = os.path.join(os.getcwd(), "llama-2-7b")
        tokenizer = LlamaTokenizer.from_pretrained(model_path, device="cuda:0")
        model = LlamaModel.from_pretrained(model_path).to("cuda:0")

        node_embeddings = get_llama_emb(
            tokenizer, model, node_text_lst, batch_size=1
        )
        class_embeddings = get_llama_emb(
            tokenizer, model, class_text_lst, batch_size=1
        )
    else:
        raise ValueError(f"Unknown language model: {name}.")

    return node_embeddings, class_embeddings


def get_llama_emb(
    tokenizer,
    model,
    text_lst,
    batch_size,
    show_progress_bar=True,
    convert_to_tensor=True,
):

    # Add EOS token for padding
    tokenizer.pad_token = tokenizer.eos_token
    all_embeddings = []

    for start_index in trange(
        0,
        len(text_lst),
        batch_size,
        desc="Batches",
        disable=not show_progress_bar,
    ):
        sentences_batch = text_lst[start_index : start_index + batch_size]
        input_ids = tokenizer(
            sentences_batch, return_tensors="pt", padding=True
        ).input_ids.to("cuda:0")
        transformer_output = model(input_ids)

        # No gradients on word_embeddings
        word_embeddings = transformer_output[0].detach()
        sentence_embeddings = word_embeddings.mean(dim=1)
        all_embeddings.extend(sentence_embeddings)

    if convert_to_tensor:
        all_embeddings = torch.stack(all_embeddings)

    # input_ids = tokenizer(text_lst, return_tensors="pt", padding=True).input_ids
    #
    # transformer_output = model(input_ids)
    #
    # # No gradients on word_embeddings
    # word_embeddings = transformer_output[0].detach()
    # sentence_embeddings = word_embeddings.mean(dim=1)

    return all_embeddings


def set_dataset_attr(dataset, name, value, size):
    dataset._data_list = None
    dataset.data[name] = value
    if dataset.slices is not None:
        dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)
