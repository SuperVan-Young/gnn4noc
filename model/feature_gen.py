import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np

class FeatureGen(nn.Module):
    """Convert embedding feature dim into global hidden dim.
    Formulation is similar to HGT.
    """
    def __init__(self, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.packet_linear = nn.Sequential(
            nn.Linear(64, h_dim),
            nn.Tanh(),
        )
        self.router_linear = nn.Sequential(
            nn.Linear(6, h_dim),
            nn.Tanh(),
        )

    def forward(self, g:dgl.heterograph):
        """Convert input feature to initial hidden feature
        """
        g.nodes['packet'].data['h'] = self.packet_linear(g.nodes['packet'].data['inp'])
        g.nodes['router'].data['h'] = self.router_linear(g.nodes['router'].data['inp'])
        