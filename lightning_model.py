import torch
from gp.lightning.module_template import BaseTemplate
from lightning.pytorch import LightningModule
import os.path as osp


class GraphPredLightning(BaseTemplate):
    def forward(self, batch):
        return self.model(batch)


class TokenPredLightning(BaseTemplate):
    def forward(self, batch):
        return self.model(batch)

    def validation_step(self, batch, batch_idx):
        tokens = self.model.generate(batch)
        with torch.no_grad():
            if self.eval_kit.has_eval_state("valid"):
                self.eval_kit.eval_step(tokens, batch, "valid")
        self.val_names = ["valid"]

    def test_step(self, batch, batch_idx):
        tokens = self.model.generate(batch)
        with torch.no_grad():
            if self.eval_kit.has_eval_state("test"):
                self.eval_kit.eval_step(tokens, batch, "test")


class GraphFinetuneLightning(BaseTemplate):
    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = self.exp_config.optimizer(
            self.exp_config.opt_params
            if self.exp_config.opt_params is not None
            else filter(lambda p: p.requires_grad, self.parameters()),
            **self.exp_config.optimizer_args
        )
        return {
            "optimizer": optimizer,
        }


class CLBaseTemplate(LightningModule):
    def __init__(
        self,
        exp_config,
        encode_model: torch.nn.Module,
        loss,
        params,
        eval_params,
        proj_head=None,
        pred_head=None,
        sup_cl=False,
        name = "",
    ):

        super().__init__()

        self.exp_config = exp_config
        self.model = encode_model
        self.loss = loss
        self.params = params
        self.eval_params = eval_params
        self.name = name
        self.proj_head = proj_head
        self.pred_head = pred_head
        self.sup_cl = sup_cl
        if self.pred_head is not None:
            self.bin_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = self.exp_config.optimizer(
            self.exp_config.opt_params
            if self.exp_config.opt_params is not None
            else self.model.parameters(),
            **self.exp_config.optimizer_args
        )
        return {
            "optimizer": optimizer,
        }

    def _calculate_cl_loss(self, g1, g2, z1, z2):

        # mask1 = g1.h_node_mask + g1.true_nodes_mask + g1.spt_nodes_mask + g1.target_node_mask
        # mask2 = g2.h_node_mask + g2.true_nodes_mask + g2.spt_nodes_mask + g2.target_node_mask
        mask1 = g1.true_nodes_mask
        mask2 = g2.true_nodes_mask
        if self.proj_head != None:
            z1 = self.proj_head(z1)[mask1]
            z2 = self.proj_head(z2)[mask2]

        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        h_node_embs = torch.stack([z1, z2], dim=1)

        # generate supervised mask for h_nodes
        # mask = torch.arange(self.params.n_way).repeat_interleave(self.params.q_query)

        # from losses import SupConLoss
        # loss = SupConLoss(temperature=0.07)
        #sup_loss = loss(features=h_node_embs, labels=mask)
        if self.sup_cl:
            print(torch.matmul(g1.bin_labels.view(-1,1), g1.bin_labels.view(1,-1)))
            cl_loss = self.loss(features=h_node_embs, mask=torch.matmul(g1.bin_labels.view(-1,1), g1.bin_labels.view(1,-1)))
        else:
            cl_loss = self.loss(features=h_node_embs, )
        return cl_loss

    def _calculate_binary_cls_loss(self, g, z):
        z = self.pred_head(z)
        score = z[g.true_nodes_mask]
        labels = g.bin_labels
        valid_ind = labels == labels
        return self.bin_loss(score.view(-1)[valid_ind], labels[valid_ind])

    def combine_loss(self, batch):
        g1, g2 = batch
        z1 = self(g1)
        z2 = self(g2)
        cl_loss = self._calculate_cl_loss(g1, g2, z1, z2)
        if self.pred_head is not None:
            binary_cls_loss = self._calculate_binary_cls_loss(g1, z1) + self._calculate_binary_cls_loss(g2, z2)
            return (cl_loss, cl_loss+binary_cls_loss, binary_cls_loss,)
        return (cl_loss, cl_loss)


    def training_step(self, batch, batch_idx):
        # get embeddings of h nodes (n_way x q_query x fs_task_num, emb_dim)
        # current only consider fs_task_num = 1
        loss = self.combine_loss(batch)
        loss_name = [osp.join(self.name, "train", "cl_loss"),
                     osp.join(self.name, "train", "loss"),
                     osp.join(self.name, "train", "pred_loss"),]
        log_dict = {loss_name[idx]: loss[idx] for idx in range(len(loss))}

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size
            if hasattr(batch, "batch_size")
            else len(batch),
        )
        return loss[1]

    def validation_step(self, batch, batch_idx):
        # get embeddings of h nodes (n_way x q_query x fs_task_num, emb_dim)
        # current only consider fs_task_num = 1
        loss = self.combine_loss(batch)
        loss_name = [osp.join(self.name, "val", "cl_loss"),
                     osp.join(self.name, "val", "loss"),
                     osp.join(self.name, "val", "pred_loss")]
        log_dict = {loss_name[idx]: loss[idx] for idx in range(len(loss))}

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size
            if hasattr(batch, "batch_size")
            else len(batch),
        )
        return loss[1]


class SimPredLightning(LightningModule):
    def __init__(
        self,
        exp_config,
        encode_model: torch.nn.Module,
        loss,
        head = None,
    ):

        super().__init__()

        self.exp_config = exp_config
        self.model = encode_model
        self.loss = loss
        self.head = head

    def forward(self, batch):
        if self.head != None:
            return self.head(self.model(batch))
        else:
            return self.model(batch)

    def configure_optimizers(self):
        optimizer = self.exp_config.optimizer(
            self.exp_config.opt_params
            if self.exp_config.opt_params is not None
            else self.model.parameters(),
            **self.exp_config.optimizer_args
        )
        return {
            "optimizer": optimizer,
        }

    def training_step(self, batch, batch_idx):
        # emb = self.batch
        # qry_emb = self.batch[batch.target_node_mask]
        print(batch.x[batch.target_node_mask])
        print(batch.x[batch.true_nodes_mask])
        print('-'*40)
        loss, acc = self.loss(self(batch)[batch.target_node_mask], self(batch)[batch.true_nodes_mask], batch.bin_labels)

        log = {"train_loss": loss, "train_acc": acc}

        self.log_dict(
            log,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size
            if hasattr(batch, "batch_size")
            else len(batch),
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss, acc = self.loss(self(batch)[batch.target_node_mask], self(batch)[batch.true_nodes_mask], batch.bin_labels)

        if dataloader_idx == 0:
            log = {"val_loss": loss, "val_acc": acc}
        elif dataloader_idx == 1:
            log = {"t_val_loss": loss, "t_val_acc": acc}

        log = {"idk_loss": loss, "idk_acc": acc}

        self.log_dict(
            log,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size
            if hasattr(batch, "batch_size")
            else len(batch),
        )

        return loss


    def test_step(self, batch, batch_idx):
        loss, acc = self.loss(self(batch)[batch.target_node_mask], self(batch)[batch.true_nodes_mask], batch.bin_labels)

        log = {"test_loss": loss, "test_acc": acc}

        self.log_dict(
            log,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size
            if hasattr(batch, "batch_size")
            else len(batch),
        )

        return loss

