import torch.nn as nn
import torch

class RBF_SVM(nn.Module):
    def __init__(self, clf):
        super().__init__()
        self.SV = nn.Parameter(torch.from_numpy(clf.support_vectors_))
        self.IC = nn.Parameter(torch.tensor(clf.intercept_))
        self.DC = nn.Parameter(torch.from_numpy(clf.dual_coef_))
        self.gamma = nn.Parameter(torch.tensor(clf.gamma))
        self.SV.requires_grad = True
        self.IC.requires_grad = True
        self.DC.requires_grad = True
        self.gamma.requires_grad = True

    def forward(self, X):
        X_expanded = X.unsqueeze(1)
        SV_broadcasted = torch.broadcast_to(self.SV, (X.shape[0], self.SV.shape[0], self.SV.shape[1]))
        res_batch = torch.subtract(SV_broadcasted, X_expanded)
        res_batch = torch.norm(res_batch, p=2, dim=2)
        res_batch = torch.square(res_batch)
        res_batch = torch.multiply(-self.gamma, res_batch)
        res_batch = torch.exp(res_batch)
        res_batch = torch.multiply(res_batch, self.DC)
        res_batch = torch.sum(res_batch, dim=1)
        res_batch = res_batch + self.IC
        return res_batch
