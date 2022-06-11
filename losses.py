"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function


import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, with_rain=None, mask=None):
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

        # for derain, concat feature and labels (which are high resolution/ no rain)
        # feature: model output
        # labels: ground truth
        # with_rain: for contrast
        # Which label (cls) to use: arbitrary, because only equal/ not equal matters

        labels_prediction = torch.zeros(features.shape[0]).cuda()
        labels_ground_truth = torch.zeros(labels.shape[0]).cuda() # prediction and gt are pushed closer
        labels_with_rain = torch.ones(with_rain.shape[0]).cuda() # prediction and with_rains are pushed further
        # import pdb; pdb.set_trace()
        features = torch.cat([features.unsqueeze(dim=1), labels.unsqueeze(dim=1), with_rain.unsqueeze(dim=1)], dim=0) # dim=0 to match label shape
        labels = torch.cat([labels_prediction, labels_ground_truth, labels_with_rain], dim=0)
        # features = (features - features.mean())/ features.std() # normalized activations ...
        # L2 normalization is assumed
        # https://github.com/HobbitLong/SupContrast/issues/65
        features = features/ torch.linalg.vector_norm(features, dim=[3,4], keepdim=True, ord=2) # ord=2 default
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
            # features = features/ torch.linalg.vector_norm(features, dim=[2], keepdim=True, ord=1) # ord=2 default

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
        exp_logits = torch.exp(logits) * logits_mask # +0.00001 # debug nan
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+0.0001)
        # import pdb; pdb.set_trace()
        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = torch.nan_to_num((mask * log_prob).sum(1) / mask.sum(1))+0.0001 # debug nan
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+0.0001) # debug nan

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        vgg = models.vgg19()
        vgg.classifier[6] = nn.Linear(4096, 2)
        # vgg.load_state_dict(torch.load('/home/peisheng/Derain/SRFBN_CVPR19/model_2.pth'), strict=False)
        vgg_pretrained_features = vgg.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class L1ContrastLoss(nn.Module):
    def __init__(self, ablation=False, is_perm=False):

        super(L1ContrastLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.ab = ablation
        self.is_perm = is_perm

    def forward(self, _a, _p, _n):
        # rename parameters to avoid collision

        a_vgg, p_vgg, n_vgg = _a, _p, _n

        if self.is_perm:
            # add permutation
            _p_perm = _p[torch.randperm(_p.size()[0])]
            _n_perm = _n[torch.randperm(_n.size()[0])]
            p_perm_vgg, n_perm_vgg = _p_perm, _n_perm

        loss = 0
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i])
            if self.is_perm:
                # add permutation
                d_ap_perm = self.l1(a_vgg[i], p_perm_vgg[i])

            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i])
                contrastive = d_ap / (d_an + 1e-7)
                if self.is_perm:
                    # add permutation
                    d_an_perm = self.l1(a_vgg[i], n_perm_vgg[i])
                    contrastive += d_ap_perm / (d_an_perm + 1e-7)
            else:
                contrastive = d_ap
                if self.is_perm:
                    contrastive += d_ap_perm

            loss += contrastive
        return loss



class ContrastLoss(nn.Module):
    def __init__(self, ablation=False, is_perm=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation
        self.is_perm = is_perm

    def forward(self, _a, _p, _n):
        # import pdb; pdb.set_trace()
        # rename parameters to avoid collision

        a_vgg, p_vgg, n_vgg = self.vgg(_a), self.vgg(_p), self.vgg(_n)

        if self.is_perm:
            # add permutation
            _p_perm = _p[torch.randperm(_p.size()[0])]
            _n_perm = _n[torch.randperm(_n.size()[0])]
            p_perm_vgg, n_perm_vgg = self.vgg(_p_perm), self.vgg(_n_perm)

        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if self.is_perm:
                # add permutation
                d_ap_perm = self.l1(a_vgg[i], p_perm_vgg[i].detach())

            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
                if self.is_perm:
                    # add permutation
                    d_an_perm = self.l1(a_vgg[i], n_perm_vgg[i].detach())
                    contrastive += d_ap_perm / (d_an_perm + 1e-7)
            else:
                contrastive = d_ap
                if self.is_perm:
                    contrastive += d_ap_perm

            loss += self.weights[i] * contrastive
        return loss



class suContrastLoss(nn.Module):
    def __init__(self, ablation=False, is_perm=True):

        super(suContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/64, 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation
        self.is_perm = is_perm

    def forward(self, _a, _p, _n):
        # import pdb; pdb.set_trace()
        # rename parameters to avoid collision

        a_vgg= self.vgg(_a)

        if self.is_perm:
            _p_perm = _p[torch.randperm(_p.size()[0])]
            _n_perm = _n[torch.randperm(_n.size()[0])]
            # add permutation
            p_perm_vgg, n_perm_vgg = self.vgg(_p_perm), self.vgg(_n_perm)

        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            if self.is_perm:
                # add permutation
                d_ap_perm = self.l1(a_vgg[i], p_perm_vgg[i].detach())

            if not self.ab:
                if self.is_perm:
                    # add permutation
                    d_an_perm = self.l1(a_vgg[i], n_perm_vgg[i].detach())
                    contrastive = d_ap_perm / (d_an_perm + 1e-7)
            else:
                contrastive = d_ap
                if self.is_perm:
                    contrastive += d_ap_perm

            loss += self.weights[i] * contrastive
        return loss




