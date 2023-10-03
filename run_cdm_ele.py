import argparse
import os
import torch
import torch_geometric as pyg
from pytorch_lightning.loggers import WandbLogger

from gp.utils.utils import (
    load_yaml,
    combine_dict,
    merge_mod,
    setup_exp,
    set_random_seed,
    k_fold2_split,
    k_fold_ind,
)
from gp.lightning.metric import (
    binary_auc_func,
    flat_binary_func,
    classification_func,
    EvalKit,
)
from gp.lightning.data_template import DataModule, DataWithMeta
from gp.lightning.training import lightning_fit, lightning_test
from gp.lightning.module_template import ExpConfig
from types import SimpleNamespace
from lightning_model import GraphPredLightning
from models.model import BinGraphModel, BinGraphAttModel, AdaPoolClassModel
from gp.nn.models.pyg import PyGGIN, PyGRGCN, PyGGINE
from models.model import PyGRGCNEdge

from torchmetrics import AUROC, Accuracy
from utils import (
    SentenceEncoder,
    binary_apr_func,
    MultiApr,
    flat_label_func,
    label_apr_func,
    binary_single_auc_func,
    binary_auc_multi_func,
    MultiAuc,
)
from gp.lightning.metric import flat_binary_func_fs
from gp.utils.utils import SmartTimer
from scipy.sparse import csr_array

from task_constructor import TaskConstructor


def main(params):
    encoder = SentenceEncoder("ST")

    tasks = TaskConstructor(["chemhiv"], encoder)

    out_dim = 768 + (params.rwpe if params.rwpe is not None else 0)
    # out_dim = 768

    if hasattr(params, "d_multiple"):
        if isinstance(params.d_multiple, str):
            data_multiple = [float(a) for a in params.d_multiple.split(",")]
        else:
            data_multiple = params.d_multiple
    else:
        data_multiple = [2, 2, 0.3, 2, 0.5, 0.4, 0.3, 2, 1, 2, 3]

    if hasattr(params, "d_min_ratio"):
        if isinstance(params.d_min_ratio, str):
            min_ratio = [float(a) for a in params.d_min_ratio.split(",")]
        else:
            min_ratio = params.d_min_ratio
    else:
        min_ratio = [0.5, 0.5, 0.05, 1, 0.1, 0.1, 0.03, 1, 0.2, 0.2, 1]

    train_data = tasks.make_train_data(data_multiple, min_ratio)

    text_dataset = tasks.make_full_dm_list(
        data_multiple, min_ratio, train_data
    )
    params.datamodule = DataModule(
        text_dataset, num_workers=params.num_workers
    )

    eval_data = text_dataset["val"] + text_dataset["test"]
    val_state = [dt.state_name for dt in text_dataset["val"]]
    test_state = [dt.state_name for dt in text_dataset["test"]]
    eval_state = val_state + test_state
    eval_metric = [dt.metric for dt in eval_data]
    eval_funcs = [dt.meta_data["eval_func"] for dt in eval_data]
    loss = torch.nn.BCEWithLogitsLoss()
    evlter = []
    for dt in eval_data:
        if dt.metric == "acc":
            evlter.append(Accuracy(task="multiclass", num_classes=dt.classes))
        elif dt.metric == "auc":
            evlter.append(AUROC(task="binary"))
        elif dt.metric == "apr":
            evlter.append(MultiApr(num_labels=dt.classes))
        elif dt.metric == "aucmulti":
            evlter.append(MultiAuc(num_labels=dt.classes))
    metrics = EvalKit(
        eval_metric,
        evlter,
        loss,
        eval_funcs,
        flat_binary_func,
        eval_mode="max",
        exp_prefix="",
        eval_state=eval_state,
        val_monitor_state=val_state[0],
        test_monitor_state=test_state[0],
    )
    # gnn = PyGGIN(params.num_layers, 768, 768)
    # gnn = PyGRGCN(params.num_layers, 3, 768, 768)
    # gnn = PyGGINE(params.num_layers, 768, 768, 768)
    gnn = PyGRGCNEdge(
        params.num_layers,
        5,
        out_dim,
        out_dim,
        drop_ratio=params.dropout,
        JK=params.JK,
    )
    bin_model = BinGraphAttModel if params.JK == "none" else BinGraphModel
    model = bin_model(gnn, out_dim, 1, add_rwpe=params.rwpe)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=params.l2
    )
    lr_scheduler = {
        "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.5),
        "interval": "epoch",
        "frequency": 1,
    }

    exp_config = ExpConfig(
        "",
        optimizer,
        dataset_callback=train_data.update,
        lr_scheduler=lr_scheduler,
    )
    exp_config.val_state_name = val_state
    exp_config.test_state_name = test_state

    pred_model = GraphPredLightning(exp_config, model, metrics)

    wandb_logger = WandbLogger(
        project=params.log_project,
        name=params.exp_name,
        save_dir=params.exp_dir,
        offline=params.offline_log,
    )

    val_res, test_res = lightning_fit(
        wandb_logger,
        pred_model,
        params.datamodule,
        metrics,
        params.num_epochs,
        save_model=False,
        load_best=False,
        reload_freq=1,
        test_rep=params.test_rep
        # profiler="simple",
        # accelerator="cpu",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rl")

    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=3)
    parser.add_argument("--q_query", type=int, default=3)
    parser.add_argument(
        "--fs_task_num",
        type=int,
        default=5,
        help="Number of tasks for few-shot training.",
    )

    params = parser.parse_args()
    configs = []
    configs.append(
        load_yaml(
            os.path.join(
                os.path.dirname(__file__), "configs", "default_config.yaml"
            )
        )
    )
    print(configs)
    # Add for few-shot parameters
    configs.append(params.__dict__)

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)
    setup_exp(mod_params)

    params = SimpleNamespace(**mod_params)
    set_random_seed(params.seed)

    torch.set_float32_matmul_precision("high")
    params.log_project = "full_cdm"
    main(params)
