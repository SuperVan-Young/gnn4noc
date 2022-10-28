import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.trace_parser.trace_parser import TraceParser

class NoCSpec():

    def __init__(self, trace_parser:TraceParser, **kwargs) -> None:
        self.trace_parser = trace_parser

        # NoC parameters
        self.core_array_h = kwargs['core_array_h']
        self.core_array_w = kwargs['core_array_w']
        self.reticle_array_h = kwargs['reticle_array_h']
        self.reticle_array_w = kwargs['reticle_array_w']

        # homogeneous bandwidth on the same level
        # normalize to core bw
        self.bandwidth = {
            'inter_core': 1,  
            'inter_reticle': kwargs['inter_reticle_bw'] // kwargs['inter_core_bw'], 
        }
        self.distance = {
            'inter_core_ew': 1,
            'inter_core_ns': 1,
            'inter_reticle_ew': 5,
            'inter_reticle_ns': 5,
        } # random distance

        # channel information
        self.channel_info = dict()
        self._setup_channel_type()

    def get_relative_bandwidth(self, s_pe, d_pe):
        if d_pe == -2:
            return 1  # eject port
        return self.channel_info[(s_pe, d_pe)]['bw']

    def _setup_channel_type(self):
        """Setup lookup table of each channel
        """
        array_w = self.core_array_w * self.reticle_array_w
        array_h = self.core_array_h * self.reticle_array_h
        xy2pe = lambda x, y: x * array_w + y  # first w, then h

        # inter core channels
        for i in range(array_w - 1):
            for j in range(array_h):
                u_pe = xy2pe(i, j)
                v_pe = xy2pe(i + 1, j)
                self.channel_info[(u_pe, v_pe)] = self.channel_info[(v_pe, u_pe)] = {
                    'bw': self.bandwidth['inter_core'],
                    'distance': self.inter_core_distance['inter_core_ew'],
                    'cnt': 0,
                }
        for i in range(array_w):
            for j in range(array_h - 1):
                u_pe = xy2pe(i, j)
                v_pe = xy2pe(i, j + 1)
                self.channel_info[(u_pe, v_pe)] = self.channel_info[(v_pe, u_pe)] = {
                    'bw': self.bandwidth['inter_core'],
                    'distance': self.inter_core_distance['inter_core_ns'],
                    'cnt': 0,
                }

        # inter reticle channels
        for i in range(1, self.reticle_array_w):
            for j in range(array_h):
                u_pe = xy2pe(self.core_array_h * i, j)
                v_pe = xy2pe(self.core_array_h * i - 1, j)
                self.channel_info[(u_pe, v_pe)] = self.channel_info[(v_pe, u_pe)] = {
                    'bw': self.bandwidth['inter_reticle'],
                    'distance': self.inter_reticle_distance['inter_reticle_ew'],
                    'cnt': 0,
                }
        for i in range(array_w):
            for j in range(1, self.reticle_array_h):
                u_pe = xy2pe(i, self.reticle_size * j)
                v_pe = xy2pe(i, self.reticle_size * j - 1)
                self.channel_info[(u_pe, v_pe)] = self.channel_info[(v_pe, u_pe)] = {
                    'bw': self.bandwidth['inter_reticle'],
                    'distance': self.inter_reticle_distance['inter_reticle_ns'],
                    'cnt': 0,
                }

    def _calculate_action_power(self, chan_info):
        """Given info of a channel, calculate the power consumption of one transfer
        """
        # for now, only a naive implementation
        return chan_info['distance'] * chan_info['cnt']

    def run(self, layer_name=None):
        G = self.trace_parser.graph_parser.get_graph(layer_name, batch=0)
        
        finished_pkts = set()
        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != 'data' or len(eattr['pkt']) == 0:
                continue
            u_pe, v_pe = G.nodes[u]['p_pe'], G.nodes[v]['p_pe']
            pid = eattr['pkt'][0]
            if pid in finished_pkts: continue  # debug: multicast pkts only count once
            finished_pkts.add(pid)

            hops = self.trace_parser.routing_parser.get_routing_hops(u_pe, v_pe, pid)
            for hop in hops:
                self.channel_info[hop]['cnt'] += eattr['cnt']

        total_power = 0
        for k, v in self.channel_info.items():
            total_power += self._calculate_action_power(v)
        return total_power

if __name__ == '__main__':
    pass
    
