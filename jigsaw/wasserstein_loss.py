import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, Function
from torchvision import datasets, transforms
import numpy



class WassersteinLossStab(Function):
    def __init__(self, cost, lam=1e-3, sinkhorn_iter=50):
        super(WassersteinLossStab, self).__init__()

        # cost = matrix M = distance matrix
        # lam = lambda of type float > 0
        # sinkhorn_iter > 0
        # diagonal cost should be 0
        self.cost = cost
        self.lam = lam
        self.sinkhorn_iter = sinkhorn_iter
        self.na = cost.size(0)
        self.nb = cost.size(1)
        self.K = torch.exp(-self.cost / self.lam)
        self.KM = self.cost * self.K
        self.stored_grad = None

    def forward(self, pred, target):
        #from jigsaw import one_hot_embedding
        #target = one_hot_embedding(labels=target, num_classes=7)

        """pred: Batch * K: K = # mass points
           target: Batch * L: L = # mass points"""
        assert pred.size(1) == self.na
        assert target.size(1) == self.nb

        batch_size = pred.size(0)

        log_a, log_b = torch.log(pred), torch.log(target)
        log_u = self.cost.new(batch_size, self.na).fill_(-numpy.log(self.na))
        log_v = self.cost.new(batch_size, self.nb).fill_(-numpy.log(self.nb))

        for i in range(self.sinkhorn_iter):
            log_u_max = torch.max(log_u, dim=1)[0]
            u_stab = torch.exp(log_u - log_u_max.expand_as(log_u))
            log_v = log_b - torch.log(torch.mm(self.K.t(), u_stab.t()).t()) - log_u_max.expand_as(log_v)
            log_v_max = torch.max(log_v, dim=1)[0]
            v_stab = torch.exp(log_v - log_v_max.expand_as(log_v))
            log_u = log_a - torch.log(torch.mm(self.K, v_stab.t()).t()) - log_v_max.expand_as(log_u)

        log_v_max = torch.max(log_v, dim=1)[0]
        v_stab = torch.exp(log_v - log_v_max.expand_as(log_v))
        logcostpart1 = torch.log(torch.mm(self.KM, v_stab.t()).t()) + log_v_max.expand_as(log_u)
        wnorm = torch.exp(log_u + logcostpart1).mean(0).sum()  # sum(1) for per item pair loss...
        grad = log_u * self.lam
        grad = grad - torch.mean(grad, dim=1).expand_as(grad)
        grad = grad - torch.mean(grad, dim=1).expand_as(grad)  # does this help over only once?
        grad = grad / batch_size

        self.stored_grad = grad

        return self.cost.new((wnorm,))

    def backward(self, grad_output):
        # print (grad_output.size(), self.stored_grad.size())
        # print (self.stored_grad, grad_output)
        res = grad_output.new()
        res.resize_as_(self.stored_grad).copy_(self.stored_grad)
        if grad_output[0] != 1:
            res.mul_(grad_output[0])
        return res, None