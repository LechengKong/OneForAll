import torch
import os.path as osp
from typing import Optional, Union, List
from lightning.pytorch import LightningModule
from gp.lightning.metric import EvalKit


class ExpConfig:
    def __init__(
        self,
        name,
        optimizer,
        opt_params=None,
        lr_scheduler=None,
        dataset_callback=None,
    ):
        self.name = name
        self.optimizer = optimizer
        self.opt_params = opt_params
        self.lr_scheduler = lr_scheduler
        self.train_state_name = ["train_eval"]
        self.val_state_name = ["valid"]
        self.test_state_name = ["test"]
        self.dataset_callback = dataset_callback

    @property
    def train_state_name(self):
        return self._train_state_name

    @property
    def val_state_name(self):
        return self._val_state_name

    @property
    def test_state_name(self):
        return self._test_state_name

    @train_state_name.setter
    def train_state_name(self, value: Union[str, List[str]]):
        if isinstance(value, str):
            self._train_state_name = [value]
        else:
            self._train_state_name = value

    @val_state_name.setter
    def val_state_name(self, value: Union[str, List[str]]):
        if isinstance(value, str):
            self._val_state_name = [value]
        else:
            self._val_state_name = value

    @test_state_name.setter
    def test_state_name(self, value: Union[str, List[str]]):
        if isinstance(value, str):
            self._test_state_name = [value]
        else:
            self._test_state_name = value

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.lr_scheduler


class BaseTemplate(LightningModule):
    def __init__(
        self,
        exp_config: ExpConfig,
        model: torch.nn.Module,
        eval_kit: Optional[EvalKit] = None,
        name: str = "",
    ):

        super().__init__()

        self.exp_config = exp_config

        self.model = model
        self.name = name

        self.eval_kit = eval_kit

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def configure_optimizers(self):
        optimizer = self.exp_config.get_optimizer()
        optimizer_dict = {"optimizer": optimizer}
        if self.exp_config.get_scheduler() is not None:
            optimizer_dict["lr_scheduler"] = self.exp_config.get_scheduler()
        return optimizer_dict

    def compute_results(
        self, batch, batch_idx, step_name, log_loss=True, *args
    ):
        try:
            score = self(batch, *args)
            loss = self.eval_kit.compute_loss(score, batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Ignoring OOM batch")
                loss = None
                score = None
            else:
                raise
        if loss is not None:
            self.log(
                osp.join(self.name, step_name, "loss"),
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=log_loss,
                batch_size=batch.batch_size
                if hasattr(batch, "batch_size")
                else len(batch),
            )
        with torch.no_grad():
            if self.eval_kit.has_eval_state(step_name):
                self.eval_kit.eval_step(score, batch, step_name)
        return score, loss

    def epoch_post_process(self, epoch_name):
        if self.eval_kit.has_eval_state(epoch_name):
            metric = self.eval_kit.eval_epoch(epoch_name)
            self.log(
                self.eval_kit.get_metric_name(epoch_name),
                metric,
                prog_bar=True,
            )
            self.eval_kit.eval_reset(epoch_name)
            return metric

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        score, loss = self.compute_results(
            batch, batch_idx, self.exp_config.train_state_name[dataloader_idx]
        )
        return loss

    def on_train_epoch_end(self):
        for name in self.exp_config.train_state_name:
            self.epoch_post_process(name)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.compute_results(
            batch,
            batch_idx,
            self.exp_config.val_state_name[dataloader_idx],
            log_loss=False,
        )

    def on_validation_epoch_end(self):
        cur_metric = []
        for name in self.exp_config.val_state_name:
            metric = self.epoch_post_process(name)
            if metric is not None:
                cur_metric.append(metric.cpu())
        if self.exp_config.dataset_callback is not None:
            self.exp_config.dataset_callback(cur_metric)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.compute_results(
            batch,
            batch_idx,
            self.exp_config.test_state_name[dataloader_idx],
            log_loss=False,
        )

    def on_test_epoch_end(self):
        for name in self.exp_config.test_state_name:
            self.epoch_post_process(name)
