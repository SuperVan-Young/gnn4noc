import os
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

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


class MessagePassing(nn.Module):
    """Passing message through a paticular edge type.
    Warning: Deprecated
    """
    def __init__(self, h_dim, etype):
        super().__init__()
        self.mlp_m = ResidualBlock(h_dim)  # message
        self.lin_r = nn.Sequential(
            nn.Linear(4*h_dim, h_dim),
            nn.LeakyReLU(0.1),
        )  # reduce
        self.etype = etype

        etype2srctype = {
            "pass": "packet",
            "transfer": "router",
            "connect": "router",
            "backpressure": "router",
        }
        etype2dsttype = {
            "pass": "router",
            "transfer": "packet",
            "connect": "router",
            "backpressure": "router",
        }
        self.srctype = etype2srctype[etype]
        self.dsttype = etype2dsttype[etype]

    def forward(self, g):
        srcfeat = g.nodes[self.srctype].data['feat']
        g.nodes[self.srctype].data['h'] = self.mlp_m(srcfeat)
        g.multi_update_all(
            {self.etype: (fn.copy_u('h', 'm'), fn.sum('m', 'h_sum'))},
            "sum"
        )
        g.multi_update_all(
            {self.etype: (fn.copy_u('h', 'm'), fn.max('m', 'h_max'))},
            "sum"
        )
        g.multi_update_all(
            {self.etype: (fn.copy_u('h', 'm'), fn.mean('m', 'h_mean'))},
            "sum"
        )
        
        h_sum = g.nodes[self.dsttype].data['h_sum']
        h_max = g.nodes[self.dsttype].data['h_max']
        h_mean = g.nodes[self.dsttype].data['h_mean']

        dstfeat = g.nodes[self.dsttype].data['feat']
        h_concat = torch.concat([h_sum, h_max, h_mean, dstfeat], dim=-1)
        dstfeat = dstfeat + self.lin_r(h_concat)  # residual connection


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
        self.stack_fn = lambda flist: torch.concat(flist, dim=-1)
        self.apply_node_fn = lambda nodes: {'feat': F.relu(nodes.data['h']) + nodes.data['feat']}

    def forward(self, g):
        g.multi_update_all({
            "pass": (fn.copy_u('feat', 'm'), fn.sum('m', 'h1')),
            "transfer": (fn.copy_u('feat', 'm'), fn.mean('m', 'h1')),
            "connect": (fn.copy_u('feat', 'm'), fn.sum('m', 'h2'))
            },
            "sum",
        )
        g.apply_nodes(lambda nodes: {'feat': 
            nodes.data['feat'] + self.lin_r(torch.concat([nodes.data['h1'], nodes.data['h2']], dim=-1))},
            ntype="router")
        g.apply_nodes(lambda nodes: {'feat':
            nodes.data['feat'] + self.lin_p(nodes.data['h1'])},
            ntype='packet')


class GraphEmbedding(nn.Module):
    """Aggregate Graph level information.
    For simplicity, we use mean reduction.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, g):
        hyper_info = torch.mean(g.nodes['packet'].data['feat'], dim=-2)
        node_info = torch.mean(g.nodes['router'].data['feat'], dim=-2)
        return torch.cat((hyper_info, node_info))


class VanillaModel(nn.Module):
    """A demo GNN model for NoC congestion prediction.
    """

    def __init__(self, h_dim=64, num_labels=4):
        super().__init__()
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

    
    def forward(self, g):
        self.feature_gen(g)
        self.conv1(g)
        self.conv2(g)
        router_embed1 = dgl.readout_nodes(g, 'feat', op='sum', ntype='router')
        router_embed2 = dgl.readout_nodes(g, 'feat', op='max', ntype='router')
        router_embed = torch.concat([router_embed1, router_embed2], dim=-1)
        pred = self.prediction_head(router_embed)
        return pred