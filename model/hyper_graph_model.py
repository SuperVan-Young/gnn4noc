from re import X
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np

from hgt import HGT, HGTLayer
from feature_gen import FeatureGen
from message_passing import MessagePassing
from dgl.nn import Set2Set

class PredictionHead(nn.Module):
    def __init__(self, h_dim=64, n_pred=2, pred_base=2.0, pred_exp_min=-1, pred_exp_max=9) -> None:
        super().__init__()
        self.set2set = Set2Set(h_dim, 1, 1)
        self.lins = nn.ModuleList()
        self.n_pred = n_pred
        for i in range(n_pred):
            lin = nn.Sequential(
                nn.Linear((2 if i == 0 else 1) * h_dim, h_dim if i != n_pred - 1 else pred_exp_max - pred_exp_min),
                nn.ELU()
            )
            self.lins.append(lin)

        self.pred_base = pred_base
        self.pred_exp_min = pred_exp_min
        self.pred_exp_max = pred_exp_max

    def forward(self, g):
        g_ = dgl.node_type_subgraph(g, ['router'])
        g_.set_batch_num_nodes(g.batch_num_nodes('router'))
        g_.set_batch_num_edges(g.batch_num_edges('connect'))
        hid = g.nodes['router'].data['h']
        x = self.set2set(g_, hid)
        for i in range(self.n_pred):
            x = self.lins[i](x)
        return x

    def congestion_to_label(self, congestion):
        ref = torch.tensor(self.pred_base ** np.arange(self.pred_exp_min, self.pred_exp_max-1), device=congestion.device).unsqueeze(0)
        label = (congestion.unsqueeze(-1) < ref).int()
        label = torch.cat([label, torch.ones(label.shape[0], 1, device=congestion.device)], dim=1)
        label = torch.argmax(label, dim=1)
        return label
    
    def label_to_congestion(self, label: int):
        return self.pred_base ** (self.pred_exp_min + label - 1) if label != 0 else 0  # left

class HyperGraphModel(nn.Module):
    """A demo GNN model for NoC congestion prediction.
    """

    def __init__(self, h_dim=64, n_hid=2, n_pred=2, message_passing="vanilla",
                 pred_base=2.0, pred_exp_min=-1, pred_exp_max=9):
        super().__init__()
        self.n_hid = n_hid

        self.feature_gen = FeatureGen(h_dim)
        self.message_passing = nn.ModuleList()
        for _ in range(n_hid):
            if message_passing == "vanilla":
                self.message_passing.append(MessagePassing(h_dim))
            elif message_passing == "HGT":
                self.message_passing.append(HGTLayer(h_dim, h_dim, 3, 7, n_heads=4, dropout=0.2, use_norm=True))
        self.prediction_head = PredictionHead(h_dim, n_pred, pred_base, pred_exp_min, pred_exp_max)
    
    def forward(self, g):
        # prepare g
        if isinstance(self.message_passing[0], HGTLayer):
            g.node_dict = dict()
            for i, ntype in enumerate(g.ntypes):
                g.node_dict[ntype] = i
                g.nodes[ntype].data['id'] = (torch.ones(g.num_nodes(ntype)).to(g.device) * i).long()
            g.edge_dict = dict()
            for i, etype in enumerate(g.etypes):
                g.edge_dict[etype] = i
                g.edges[etype].data['id'] = (torch.ones(g.num_edges(etype)).to(g.device) * i).long()

        self.feature_gen(g)
        for i in range(self.n_hid):
            self.message_passing[i](g, 'h', 'h')
        pred = self.prediction_head(g)
        return pred

    