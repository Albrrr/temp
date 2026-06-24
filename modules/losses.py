import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse

def isnan(x):
    return x != x

def mean(l, ignore_nan=False, empty=0):
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probas, labels, classes='present'):
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

def flatten_probas(probas, labels, ignore=None):
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(LovaszSoftmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        if self.per_image:
            loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), self.ignore), classes=self.classes) for prob, lab in zip(probas, labels))
        else:
            loss = lovasz_softmax_flat(*flatten_probas(probas, labels, self.ignore), classes=self.classes)
        return loss

def one_hot(label, n_classes, requires_grad=True):
    one_hot_label = torch.eye(n_classes, device=label.device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)
    return one_hot_label

class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        n, c, _, _ = pred.shape
        one_hot_gt = one_hot(gt, c)
        gt_b = F.max_pool2d(1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt
        pred_b = F.max_pool2d(1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        P = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)
        BF1 = 2 * P * R / (P + R + 1e-7)
        loss = torch.mean(1 - BF1)
        return loss

class DeCovLoss(nn.Module):
    def __init__(self):
        super(DeCovLoss, self).__init__()

    def forward(self, x):
        N = x.size(0)
        if N <= 1:
            return torch.tensor(0.0, device=x.device)
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.t() @ x) / (N - 1)
        cov_diag = torch.diag(cov)
        loss = 0.5 * (cov.pow(2).sum() - cov_diag.pow(2).sum())
        return loss

class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine, labels):
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        loss = F.cross_entropy(output, labels)
        return loss
class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, embeddings, labels):
        sim_mat = torch.matmul(embeddings, embeddings.t())
        
        label_mat = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        pos_mask = label_mat & ~eye
        neg_mask = ~label_mat

        ap = torch.clamp_min(-sim_mat.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sim_mat.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sim_mat - delta_p) * self.gamma
        logit_n = an * (sim_mat - delta_n) * self.gamma

        logit_p = logit_p.masked_fill(~pos_mask, float('-inf'))
        logit_n = logit_n.masked_fill(~neg_mask, float('-inf'))

        lse_p = torch.logsumexp(logit_p, dim=1)
        lse_n = torch.logsumexp(logit_n, dim=1)

        loss = self.soft_plus(lse_p + lse_n)
        
        valid = (pos_mask.sum(1) > 0) & (neg_mask.sum(1) > 0)
        if valid.sum() > 0:
            return loss[valid].mean()
        else:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
