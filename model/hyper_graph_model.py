from inspect import stack
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np

from dgl.nn import SumPooling, AvgPooling, MaxPooling, Set2Set

def get_activation_func(name):
    if name == "ReLU":
        return nn.ReLU()
    elif name == "LeakyReLU":
        return nn.LeakyReLU()
    elif name == "ELU":
        return nn.ELU()
    else:
        raise NotImplementedError

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation="ReLU"):
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim)
        self.activation = get_activation_func(activation)
    
    def forward(self, x):
        return self.activation(self.lin(x))


class FeatureGen(nn.Module):
    """Convert embedding feature dim into global hidden dim.
    """
    def __init__(self, h_dim, activation="ReLU"):
        super().__init__()
        self.h_dim = h_dim
        self.packet_freq = LinearBlock(1, h_dim, activation)
        self.packet_flit = LinearBlock(32, h_dim, activation)
        self.router_op_type = LinearBlock(4, h_dim, activation)
        self.channel_bandwidth = LinearBlock(1, h_dim, activation)
        self.fuse_packet = nn.Linear(2 * h_dim, h_dim)
        self.fuse_router = nn.Linear(h_dim, h_dim)
        self.fuse_channel = nn.Linear(h_dim, h_dim)

    def forward(self, g:dgl.heterograph):
        """Return generated feature of each type of node.
        """
        freq_feat = self.packet_freq(g.nodes['packet'].data['freq'])
        flit_feat = self.packet_flit(g.nodes['packet'].data['flit'])
        packet_feat = torch.concat([freq_feat, flit_feat], dim=1)
        packet_feat = self.fuse_packet(packet_feat)

        op_type_feat = self.router_op_type(g.nodes['router'].data['op_type'])
        router_feat = self.fuse_router(op_type_feat)

        bandwidth_feat = self.channel_bandwidth(g.nodes['channel'].data['bandwidth'])
        channel_feat = self.fuse_channel(bandwidth_feat)

        return {
            "packet": packet_feat,    # (#packet, h_dim)
            "router": router_feat,    # (#router, h_dim)
            "channel": channel_feat,  # (#channel, h_dim)
        }


class MessagePassing(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        assert h_dim % 2 == 0
        self.h_dim = h_dim
        self.lin = nn.Linear(h_dim, (h_dim ** 2) // 2)
            

    def forward(self, g:dgl.heterograph, packet_feat, router_feat):
        g.nodes['packet'].data['pfeat'] = self.lin(packet_feat).reshape(-1, self.h_dim, self.h_dim // 2)
        g.nodes['router'].data['rfeat'] = router_feat.unsqueeze(-1)

        # channel gather router feature
        g.multi_update_all({
            "output": (fn.copy_u('rfeat', 'm'), fn.sum('m', 'rfeat_in')),     # generates m_in
            "input_inv": (fn.copy_u('rfeat', 'm'), fn.sum('m', 'rfeat_out'))  # generates m_out
            }, "sum",
        )

        # channel generate message w.r.t. packets passing it
        g.multi_update_all({"pass": (fn.u_mul_v('pfeat', 'rfeat_in', 'm'), fn.sum('m', 'partial_m_in'))},"sum")
        g.multi_update_all({"pass": (fn.u_mul_v('pfeat', 'rfeat_out', 'm'), fn.sum('m', 'partial_m_out'))},"sum")
        g.nodes['channel'].data['partial_m_in'] = g.nodes['channel'].data['partial_m_in'].sum(dim=1)  # (channel, h_dim // 2)
        g.nodes['channel'].data['partial_m_out'] = g.nodes['channel'].data['partial_m_out'].sum(dim=1)  # (channel, h_dim // 2)

        # router gather partial message from channel
        g.multi_update_all({
            "input": (fn.copy_u('partial_m_in', 'm'), fn.sum('m', 'm_in')),
            "output_inv": (fn.copy_u('partial_m_out', 'm'), fn.sum('m', 'm_out')),
            }, "sum"
        )

        m_in, m_out = g.nodes['router'].data['m_in'], g.nodes['router'].data['m_out']
        m = torch.concat([m_in, m_out], dim=1)  # (#router, h_dim)

        return m


class ActivateUpdater(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = get_activation_func(activation)

    def forward(self, message, feat):
        return self.activation(feat + message)

class HyperGraphModel(nn.Module):
    """A demo GNN model for NoC congestion prediction.
    """

    def __init__(self, h_dim=64, activation='ReLU', num_mp=2, update="activate", readout="sum", pred_layer=2,
                 pred_base=2.0, pred_exp_min=-2, pred_exp_max=9):
        super().__init__()
        self.num_mp = num_mp          # number of message passing iteration
        self.pred_base = pred_base
        self.pred_exp_min = pred_exp_min
        self.pred_exp_max = pred_exp_max

        
        self.activation = get_activation_func(activation)

        self.feature_gen = FeatureGen(h_dim, activation)
        
        self.message_passing = MessagePassing(h_dim)

        if update == "GRU":
            self.updater = nn.GRUCell(h_dim, h_dim)
        elif update == "activate":
            self.updater = ActivateUpdater(activation)
        else:
            raise NotImplementedError

        if readout == "set2set":
            self.graph_pooling = Set2Set(h_dim, 1, 1)
        elif readout == "sum":
            self.graph_pooling = SumPooling()
        elif readout == "avg":
            self.graph_pooling = AvgPooling()
        elif readout == "max":
            self.graph_pooling = MaxPooling()

        prediction_layers = []
        assert pred_layer >= 0
        for i in range(pred_layer):
            prediction_layers.append(nn.Linear(h_dim, h_dim)),
            prediction_layers.append(get_activation_func(activation)),
        prediction_layers.append(nn.Linear(h_dim, pred_exp_max - pred_exp_min))
        self.prediction_head = nn.Sequential(*prediction_layers)

        self.__ref_labels = torch.tensor(pred_base ** np.arange(pred_exp_min, pred_exp_max-1))
    
    def forward(self, g):
        feats = self.feature_gen(g)
        hidden = feats['router']
        for i in range(self.num_mp):
            message = self.message_passing(g, feats['packet'], hidden)
            hidden = self.updater(message, hidden)
        g.nodes['router'].data['feat'] = hidden
        # a very ugly work-around for batched heterogeneous graph
        graph_embeds = []
        for subgraph in dgl.unbatch(g):
            sub_g = dgl.graph(([], []), num_nodes=subgraph.num_nodes(ntype='router'))
            sub_hid = subgraph.nodes['router'].data['feat']
            graph_embeds.append(self.graph_pooling(sub_g, sub_hid))
        graph_embed = torch.concat(graph_embeds, dim=0)
        pred = self.prediction_head(graph_embed)
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
        return self.pred_base ** (self.pred_exp_min + label - 1) if label != 0 else 0  # left