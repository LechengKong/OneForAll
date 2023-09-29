import torch
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint

from gp.utils.utils import dict_res_summary
from gp.lightning.metric import EvalKit


def lightning_fit(
    logger,
    model,
    data_module,
    metrics: EvalKit,
    num_epochs,
    profiler=None,
    cktp_prefix="",
    load_best=True,
    prog_freq=20,
    test_rep=1,
    save_model=True,
    prog_bar=True,
    accelerator="auto",
    detect_anomaly=False,
    reload_freq=0,
    val_interval=1,
):
    callbacks = []
    if prog_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=20))
    if save_model:
        callbacks.append(
            ModelCheckpoint(
                monitor=metrics.val_metric,
                mode=metrics.eval_mode,
                save_last=True,
                filename=cktp_prefix + "{epoch}-{step}",
            )
        )
    trainer = Trainer(
        accelerator=accelerator,
        devices=1 if torch.cuda.is_available() else 10,
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=prog_freq,
        profiler=profiler,
        enable_checkpointing=save_model,
        enable_progress_bar=prog_bar,
        detect_anomaly=detect_anomaly,
        reload_dataloaders_every_n_epochs=reload_freq,
        check_val_every_n_epoch=val_interval,
    )
    trainer.fit(model, datamodule=data_module)
    if load_best:
        model.load_state_dict(
            torch.load(trainer.checkpoint_callback.best_model_path)[
                "state_dict"
            ]
        )

    val_col = []
    for i in range(test_rep):
        val_col.append(
            trainer.validate(model, datamodule=data_module, verbose=False)[0]
        )

    val_res = dict_res_summary(val_col)
    for met in val_res:
        val_mean = np.mean(val_res[met])
        val_std = np.std(val_res[met])
        print("{}:{:f}±{:f}".format(met, val_mean, val_std))

    target_val_mean = np.mean(val_res[metrics.val_metric])
    target_val_std = np.std(val_res[metrics.val_metric])

    test_col = []
    for i in range(test_rep):
        test_col.append(
            trainer.test(model, datamodule=data_module, verbose=False)[0]
        )

    test_res = dict_res_summary(test_col)
    for met in test_res:
        test_mean = np.mean(test_res[met])
        test_std = np.std(test_res[met])
        print("{}:{:f}±{:f}".format(met, test_mean, test_std))

    target_test_mean = np.mean(test_res[metrics.test_metric])
    target_test_std = np.std(test_res[metrics.test_metric])
    return [target_val_mean, target_val_std], [
        target_test_mean,
        target_test_std,
    ]


def lightning_test(
    logger,
    model,
    data_module,
    metrics: EvalKit,
    model_dir: str,
    profiler=None,
    prog_freq=20,
    test_rep=1,
    prog_bar=True,
    accelerator="auto",
    detect_anomaly=False,
):
    callbacks = []
    if prog_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=20))
    trainer = Trainer(
        accelerator=accelerator,
        devices=1 if torch.cuda.is_available() else 10,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=prog_freq,
        profiler=profiler,
        enable_progress_bar=prog_bar,
        detect_anomaly=detect_anomaly,
    )
    # trainer.fit(model, datamodule=data_module)
    model.load_state_dict(torch.load(model_dir)["state_dict"])

    val_col = []
    for i in range(test_rep):
        val_col.append(
            trainer.validate(model, datamodule=data_module, verbose=False)[0]
        )

    val_res = dict_res_summary(val_col)
    for met in val_res:
        val_mean = np.mean(val_res[met])
        val_std = np.std(val_res[met])
        print("{}:{:f}±{:f}".format(met, val_mean, val_std))

    target_val_mean = np.mean(val_res[metrics.val_metric])
    target_val_std = np.std(val_res[metrics.val_metric])

    test_col = []
    for i in range(test_rep):
        test_col.append(
            trainer.test(model, datamodule=data_module, verbose=False)[0]
        )

    test_res = dict_res_summary(test_col)
    for met in test_res:
        test_mean = np.mean(test_res[met])
        test_std = np.std(test_res[met])
        print("{}:{:f}±{:f}".format(met, test_mean, test_std))

    target_test_mean = np.mean(test_res[metrics.test_metric])
    target_test_std = np.std(test_res[metrics.test_metric])
    return [target_val_mean, target_val_std], [
        target_test_mean,
        target_test_std,
    ]
