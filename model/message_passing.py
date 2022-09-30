import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np

class MessagePassing(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        assert h_dim % 2 == 0
        self.h_dim = h_dim
        self.lin_p = nn.Linear(h_dim, (h_dim ** 2) // 2)
        self.lin_c = nn.Linear(2 * h_dim, h_dim)
        self.norm_r = nn.LayerNorm(h_dim)
        self.norm_p = nn.LayerNorm(h_dim)
        self.skip_r = nn.Parameter(torch.ones(1))  # learnable residual connection
        self.skip_p = nn.Parameter(torch.ones(1))
        self.drop = nn.Dropout(0.2)

    def forward(self, g:dgl.heterograph, inp_key=None, out_key=None):
        # same interface as HGT

        # channel gather router hidden state
        g.multi_update_all({
            "output": (fn.copy_u('h', 'm'), fn.sum('m', 'h_in')),     # generates m_in
            "input_inv": (fn.copy_u('h', 'm'), fn.sum('m', 'h_out'))  # generates m_out
            }, "sum",
        )
        g.nodes['channel'].data['h_in'] = g.nodes['channel'].data['h_in'].unsqueeze(-1)
        g.nodes['channel'].data['h_out'] = g.nodes['channel'].data['h_out'].unsqueeze(-1)

        # channel generate message w.r.t. packets passing it
        g.nodes['packet'].data['e'] = self.lin_p(g.nodes['packet'].data['h']).reshape(-1, self.h_dim, self.h_dim // 2)
        g.multi_update_all({"pass": (fn.u_mul_v('e', 'h_in', 'm'), fn.sum('m', 'm_in'))},"sum")
        g.multi_update_all({"pass": (fn.u_mul_v('e', 'h_out', 'm'), fn.sum('m', 'm_out'))},"sum")
        g.nodes['channel'].data['m_in'] = g.nodes['channel'].data['m_in'].sum(dim=1)  # (channel, h_dim // 2)
        g.nodes['channel'].data['m_out'] = g.nodes['channel'].data['m_out'].sum(dim=1)  # (channel, h_dim // 2)

        # router gather partial message from channel
        g.multi_update_all({
            "input": (fn.copy_u('m_in', 'm'), fn.sum('m', 'm_in')),
            "output_inv": (fn.copy_u('m_out', 'm'), fn.sum('m', 'm_out')),
            }, "sum"
        )
        m_in, m_out = g.nodes['router'].data['m_in'], g.nodes['router'].data['m_out']
        m = torch.relu(torch.concat([m_in, m_out], dim=1))

        # router update hidden state
        alpha_r = torch.sigmoid(self.skip_r)
        g.nodes['router'].data['h'] = m * alpha_r + g.nodes['router'].data['h'] * (1 - alpha_r)
        g.nodes['router'].data['h'] = self.drop(self.norm_r(g.nodes['router'].data['h']))

        # packet gather all channels information
        g.nodes['channel'].data['c'] = torch.concat([g.nodes['channel'].data['h_in'], g.nodes['channel'].data['h_out']], dim=1).squeeze(-1)
        g.multi_update_all({'pass_inv': (fn.copy_u('c', 'm'), fn.mean('m', 'c'))}, 'sum')
        c = torch.relu(self.lin_c(g.nodes['packet'].data['c']))
        
        # packet update hidden state
        alpha_p = torch.sigmoid(self.skip_p)
        g.nodes['packet'].data['h'] = c * alpha_p + g.nodes['packet'].data['h'] * (1 - alpha_p)
        g.nodes['packet'].data['h'] = self.drop(self.norm_p(g.nodes['packet'].data['h']))

        return m