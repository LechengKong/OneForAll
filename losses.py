from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import Module


class SupConLoss(Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_way):
        super(PrototypicalLoss, self).__init__()
        self.n_way = n_way

    def euclidean_dist(self, x, y):
        '''
            Compute euclidean distance between two tensors
            '''
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x - y, 2).sum(2)

    def forward(self, qry_emb, prompt_emb, bin_labels):
        print(qry_emb)
        print(prompt_emb)
        dists = self.euclidean_dist(qry_emb, prompt_emb)
        # Extract required slices based on your loop logic
        sliced_dists = [dists[i, i * self.n_way: (i + 1) * self.n_way] for i in range(qry_emb.size(0))]


        concat_dists = torch.stack(sliced_dists, dim=0)
        # Compute loss using a vectorized approach
        log_probs = F.log_softmax(-concat_dists, dim=1)
        loss = -(log_probs * bin_labels.view(-1,self.n_way)).sum() / qry_emb.size(0)

        _, y_hat = log_probs.max(1)
        # print(log_probs)
        # print(y_hat)
        #print(bin_labels.view(-1,self.n_way))
        acc_val = y_hat.eq(torch.argmax(bin_labels, dim=-1)).float().mean()

        return loss, acc_val



# def euclidean_dist(x, y):
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     if d != y.size(1):
#         raise Exception
#     x = x.unsqueeze(1).expand(n, m, d)
#     y = y.unsqueeze(0).expand(n, m, d)
#     return torch.pow(x - y, 2).sum(2)

# for i in range(a.size(0)):
#     aa = a[i,:].view(1,-1)
#     bb = b[i * n_way:(i+1) *n_way, :]
#     dd = euclidean_dist(aa, bb)
#     loss = (F.log_softmax(-dd, dim=1) * bin_labels[i * n_way: (i+1) * n_way]).sum()
#     losses += loss
#     _, y_hat = log_p_y.max(2)
#     acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
#
#     return loss_val, acc_val
#
#
# def prototypical_loss(qry_emb, bin_labels, prompt_emb):
#     '''
#     Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
#
#     Compute the barycentres by averaging the features of n_support
#     samples for each class in target, computes then the distances from each
#     samples' features to each one of the barycentres, computes the
#     log_probability for each n_query samples for each one of the current
#     classes, of appartaining to a class c, loss and accuracy are then computed
#     and returned
#     Args:
#     - input: the model output for a batch of samples
#     - target: ground truth for the above batch of samples
#     - n_support: number of samples to keep in account when computing
#       barycentres, for each one of the current classes
#     '''
#     target_cpu = target.to('cpu')
#     input_cpu = input.to('cpu')
#
#     def supp_idxs(c):
#         # FIXME when torch will support where as np
#         return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)
#
#     # FIXME when torch.unique will be available on cuda too
#     classes = torch.unique(target_cpu)
#     n_classes = len(classes)
#     # FIXME when torch will support where as np
#     # assuming n_query, n_target constants
#     n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
#
#     support_idxs = list(map(supp_idxs, classes))
#
#     prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
#     # FIXME when torch will support where as np
#     query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
#
#     query_samples = input.to('cpu')[query_idxs]
#     dists = euclidean_dist(query_samples, prototypes)
#
#     log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
#
#     target_inds = torch.arange(0, n_classes)
#     target_inds = target_inds.view(n_classes, 1, 1)
#     target_inds = target_inds.expand(n_classes, n_query, 1).long()
#
#     loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
#     _, y_hat = log_p_y.max(2)
#     acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
#
#     return loss_val,  acc_val