import os
import sys
from tkinter.tix import Tree
from matplotlib.pyplot import autoscale
import numpy as np
from tqdm import tqdm
from scipy.optimize import linprog
import networkx as nx
sys.path.append("..")

import global_control as gc
from focus_agent.focus_agent import FocusAgent
from trace_parser.trace_parser import TraceParser

class NoCPowerPredictor():

    def __init__(self, trace_parser:TraceParser, reticle_size, die_size=None) -> None:
        self.trace_parser = trace_parser
        self.G = None
        self.array_size = trace_parser.spec_parser.get_array_size()

        # NoC parameters
        # core -> reticle -> die
        self.reticle_size = reticle_size
        assert self.array_size % self.reticle_size == 0
        self.die_size = die_size
        if die_size is not None:
            assert self.array_size % die_size == 0

        self.inter_core_distance = 1  # random
        self.inter_reticle_distance = 5  # random
        self.inter_die_distance = 10  # random

        # channel information
        self.channel_info = dict()
        self._setup_channel_type()

    def _setup_channel_type(self):
        """Setup lookup table of each channel
        """
        xy2pe = lambda x, y: x * self.array_size + y

        # inter core channels
        for i in range(self.array_size - 1):
            for j in range(self.array_size):
                u_pe = xy2pe(i, j)
                v_pe = xy2pe(i + 1, j)
                self.channel_info[(u_pe, v_pe)] = self.channel_info[(v_pe, u_pe)] = {
                    'type': 'inter_core',
                    'distance': self.inter_core_distance,
                    'cnt': 0,
                }
        for i in range(self.array_size):
            for j in range(self.array_size - 1):
                u_pe = xy2pe(i, j)
                v_pe = xy2pe(i, j + 1)
                self.channel_info[(u_pe, v_pe)] = self.channel_info[(v_pe, u_pe)] = {
                    'type': 'inter_core',
                    'distance': self.inter_core_distance,
                    'cnt': 0,
                }

        # inter reticle channels
        for i in range(1, self.array_size // self.reticle_size):
            for j in range(self.array_size):
                u_pe = xy2pe(self.reticle_size * i, j)
                v_pe = xy2pe(self.reticle_size * i - 1, j)
                self.channel_info[(u_pe, v_pe)] = self.channel_info[(v_pe, u_pe)] = {
                    'type': 'inter_reticle',
                    'cnt': 0,
                    'distance': self.inter_reticle_distance,
                }
        for i in range(self.array_size):
            for j in range(1, self.array_size // self.reticle_size):
                u_pe = xy2pe(i, self.reticle_size * j)
                v_pe = xy2pe(i, self.reticle_size * j - 1)
                self.channel_info[(u_pe, v_pe)] = self.channel_info[(v_pe, u_pe)] = {
                    'type': 'inter_reticle',
                    'cnt': 0,
                    'distance': self.inter_reticle_distance,
                }
        
        # inter die channels (cover inter reticle channels, optional)
        if self.die_size is not None:
            for i in range(1, self.array_size // self.die_size):
                for j in range(self.array_size):
                    u_pe = xy2pe(self.die_size * i, j)
                    v_pe = xy2pe(self.die_size * i - 1, j)
                    self.channel_info[(u_pe, v_pe)] = self.channel_info[(v_pe, u_pe)] = {
                        'type': 'inter_die',
                        'cnt': 0,
                        'distance': self.inter_die_distance,
                    }
            for i in range(self.array_size):
                for j in range(1, self.array_size // self.die_size):
                    u_pe = xy2pe(i, self.die_size * j)
                    v_pe = xy2pe(i, self.die_size * j - 1)
                    self.channel_info[(u_pe, v_pe)] = self.channel_info[(v_pe, u_pe)] = {
                        'type': 'inter_die',
                        'cnt': 0,
                        'distance': self.inter_die_distance,
                    }

    def _calculate_action_power(self, chan_info):
        """Given info of a channel, calculate the power consumption of one transfer
        """
        # for now, only a naive implementation
        return chan_info['distance'] * chan_info['cnt']

    def run(self, layer_name=None):
        G = self.G = self.trace_parser.graph_parser.get_graph(layer_name, batch=0)
        
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
                self.channel_info[hop]['cnt'] += 1

        total_power = 0
        for k, v in self.channel_info.items():
            total_power += self._calculate_action_power(v)
        return total_power


def test_example():
    agent = FocusAgent(fake_trace=True, simulate=True)
    taskname ="cw64_ci64_co64_bw0_bi1_fw2_fi7_fo12_dw1_di1_do1_n8" + f"_b1w1024_8x8"
    trace_parser = TraceParser(
        agent.get_op_graph_path(taskname),
        agent.get_outlog_path(taskname),
        agent.get_routing_path(taskname),
        agent.get_spec_path(taskname),
    )

    predictor = NoCPowerPredictor(trace_parser, reticle_size=8, die_size=None)
    result = predictor.run()
    print(result)


if __name__ == '__main__':
    test_example()
    
