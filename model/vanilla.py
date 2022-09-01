import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import NNConv

class VanillaModel(nn.Module):
    """A demo model with 2-layer NNConv and 2-layer MLP
    """

    def __init__(self, n_feats, e_feats, h_feats):
        super().__init__()
        self.elin_1 = nn.Linear(e_feats, n_feats * h_feats)
        self.elin_2 = nn.Linear(e_feats, h_feats * h_feats)
        self.conv1 = NNConv(n_feats, h_feats, lambda x: self.elin_1(x))
        self.conv2 = NNConv(h_feats, h_feats, lambda x: self.elin_2(x))
        # self.inlat_lin_1 = nn.Linear(h_feats, h_feats)
        self.inlat_lin_2 = nn.Linear(h_feats, 1)


    def forward(self, g, nfeat, efeat):
        h = self.conv1(g, nfeat, efeat)
        h = F.relu(h)
        h = self.conv2(g, h, efeat)
        h = F.relu(h)
        # h = self.inlat_lin_1(h)
        # h = F.relu(h)
        h = self.inlat_lin_2(h)

        return h