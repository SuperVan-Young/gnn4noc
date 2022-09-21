import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.lin(x)


class ResidualBlock(nn.Module):
    def __init__(self, h_dim, input_dim=None):
        super().__init__()
        if input_dim == None:
            input_dim = h_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.mlp(x)


class FeatureGen(nn.Module):
    """Convert embedding feature dim into global hidden dim.
    """
    def __init__(self, h_dim):
        super().__init__()
        self.hyper_lin_freq = LinearBlock(1, h_dim)
        self.hyper_lin_flit = LinearBlock(32, h_dim)
        self.node_lin_op_type = LinearBlock(4, h_dim)
        self.fuse_hyper = LinearBlock(2*h_dim, h_dim)
        self.fuse_node = LinearBlock(h_dim, h_dim)

    def forward(self, g:dgl.heterograph):
        freq_feat = self.hyper_lin_freq(g.nodes['packet'].data['freq'])
        flit_feat = self.hyper_lin_flit(g.nodes['packet'].data['flit'])
        hyper_node_feat = torch.concat([freq_feat, flit_feat], dim=1)
        hyper_node_feat = self.fuse_hyper(hyper_node_feat)

        op_type_feat = self.node_lin_op_type(g.nodes['router'].data['op_type'])
        node_feat = self.fuse_node(op_type_feat)

        g.nodes['router'].data['feat'] = node_feat
        g.nodes['packet'].data['feat'] = hyper_node_feat


class HeteroGraphConv(nn.Module):
    """Simplified version of the above message passing"""
    def __init__(self, h_dim):
        super().__init__()
        self.lin_r = nn.Sequential(
            nn.Linear(2*h_dim, h_dim),
            nn.ReLU()
        )
        self.lin_p = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

    def forward(self, g):
        g.multi_update_all({
            "pass": (fn.copy_u('feat', 'm'), fn.sum('m', 'h1')),
            "transfer": (fn.copy_u('feat', 'm'), fn.mean('m', 'h1')),
            "connect": (fn.copy_u('feat', 'm'), fn.sum('m', 'h2'))
            },
            "sum",
        )
        g.apply_nodes(self.__agg_r, ntype="router")
        g.apply_nodes(self.__agg_p, ntype='packet')

    def __agg_r(self, nodes):
        agged = torch.concat([nodes.data['h1'], nodes.data['h2']], dim=-1)
        return {'feat': nodes.data['feat'] + self.lin_r(agged)}

    def __agg_p(self, nodes):
        agged = nodes.data['h1']
        return {'feat': nodes.data['feat'] + self.lin_p(agged)}


class VanillaModel(nn.Module):
    """A demo GNN model for NoC congestion prediction.
    """

    def __init__(self, h_dim=64, base=2.0, label_min=-2, label_max=9):
        super().__init__()
        self.base = base
        self.label_min = label_min
        self.label_max = label_max
        num_labels = label_max - label_min
        self.feature_gen = FeatureGen(h_dim)
        self.conv1 = HeteroGraphConv(h_dim)
        self.conv2 = HeteroGraphConv(h_dim)
        self.prediction_head = nn.Sequential(
            nn.Linear(2*h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_labels)
        )
        self.__ref_labels = torch.tensor(base ** np.arange(label_min, label_max-1))

    
    def forward(self, g):
        self.feature_gen(g)
        self.conv1(g)
        self.conv2(g)
        router_embed1 = dgl.readout_nodes(g, 'feat', op='sum', ntype='router')
        router_embed2 = dgl.readout_nodes(g, 'feat', op='max', ntype='router')
        router_embed = torch.concat([router_embed1, router_embed2], dim=-1)
        pred = self.prediction_head(router_embed)
        return pred

    def congestion_to_label(self, congestion):
        if isinstance(congestion, float):
            label = (congestion < self.__ref_labels).int()
            label = torch.cat([label, torch.ones(1,)])
            label = torch.argmax(label)
        elif isinstance(congestion, torch.Tensor):
            label = (congestion.unsqueeze(-1) < self.__ref_labels.unsqueeze(0)).int()
            label = torch.cat([label, torch.ones(label.shape[0], 1,)], dim=1)
            label = torch.argmax(label, dim=1)
        return label
    
    def label_to_congestion(self, label: int):
        return self.base ** (self.label_min + label - 1) if label != 0 else 0  # left