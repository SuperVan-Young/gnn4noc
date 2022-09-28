import os
import sys
sys.path.append("..")

import networkx as nx
import torch
import dgl

from base import SmartDict, binarize_float
from hyper import HyperGraphGenerator

class OutputPortGraphGenerator(HyperGraphGenerator):

    def _gen_graph(self, G: nx.DiGraph):
        graph, pkt2id, rt2id, chan2id = self._gen_skeleton(G)
        for attr, t in self._gen_packet_attr(G, pkt2id).items():
            graph.nodes['packet'].data[attr] = t
        # actually it's port, but I call it router anyway, so that I don't have to write another model
        for attr, t in self._gen_router_attr(G, rt2id).items():
            graph.nodes['router'].data[attr] = t
        # virtual channel shouldn't carry information

        return graph

    def _get_port(self, src_router, dst_router):
        """For the output port from src router to dst router, generate a unique port number.
        Returns: int
        """
        # for every router, it has <= 4 NoC port (0,1,2,3), a injection port (4), a ejection port (5)
        directions = {
            (0, 1): 0, # north
            (1, 0): 1, # east
            (0, -1): 2, # south
            (-1, 0): 3, # west
            'inject': 4,
            'eject': 5,
        }
        N = len(directions)

        if src_router == -1:  # sender is core
            return N * dst_router + directions['inject']
        if dst_router == -2:  # receiver is core
            return N * src_router + directions['eject']
        assert src_router >= 0
        assert dst_router >= 0

        k = self.parser.spec_parser.get_array_size()
        sx, sy = src_router // k, src_router % k
        dx, dy = dst_router // k, dst_router % k
        return N * src_router + directions[(dx-sx, dy-sy)]


    def _gen_skeleton(self, G:nx.DiGraph):
        """Similar to Hyper Graph Generator
        """
        packet_to_channel = dict()  # pid: cids
        port_to_channel_in = dict()  # rid: cids
        port_to_channel_out = dict()  # rid: cids

        pkt2id = SmartDict()
        rt2id = SmartDict()
        chan2id = SmartDict()

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != "data" or len(eattr["pkt"]) == 0:
                continue  # could happen when insrc and worker on the same pe

            pkt = eattr["pkt"][0]
            u_pe, v_pe = G.nodes[u]['p_pe'], G.nodes[v]['p_pe']
            channels = self.parser.routing_parser.get_routing_hops(u_pe, v_pe, pkt)

            w = -1  # previous port, started with a magic number for NI of source core
            _, dest_port = channels[-1]
            channels.append((dest_port, -2))  # magic number for NI of dest core

            pid = pkt2id[pkt]
            packet_to_channel[pid] = []
            for s, d in channels:
                port_w = self._get_port(w, s)
                port_s = self._get_port(s, d)
                port_w_id = rt2id[port_w]
                port_s_id = rt2id[port_s]
                w = s  # update previous port

                cid = chan2id[(port_w_id, port_s_id)]

                packet_to_channel[pid].append(cid)        # packet pass this channel
                if port_w_id not in port_to_channel_out.keys():
                    port_to_channel_out[port_w_id] = []
                if cid not in port_to_channel_out[port_w_id]:
                    port_to_channel_out[port_w_id].append(cid)  # this channel is source port's output channel
                if port_s_id not in port_to_channel_in.keys():
                    port_to_channel_in[port_s_id] = []
                if cid not in port_to_channel_in[port_s_id]:
                    port_to_channel_in[port_s_id].append(cid)   # this channel is destination port's input channel

        pass_srcs, pass_dsts = self.__parse_edges_dict(packet_to_channel)
        pass_inv_srcs, pass_inv_dsts = pass_dsts.clone().detach(), pass_srcs.clone().detach()
        output_srcs, output_dsts = self.__parse_edges_dict(port_to_channel_out)
        output_inv_srcs, output_inv_dsts = output_dsts.clone().detach(), output_srcs.clone().detach()
        input_dsts, input_srcs = self.__parse_edges_dict(port_to_channel_in)
        input_inv_srcs, input_inv_dsts = input_dsts.clone().detach(), input_srcs.clone().detach()

        graph = dgl.heterograph({
            ('packet', 'pass', 'channel'): (pass_srcs, pass_dsts),
            ('channel', 'pass_inv', 'packet'): (pass_inv_srcs, pass_inv_dsts),
            ('router', 'output', 'channel'): (output_srcs, output_dsts),
            ('channel', 'output_inv', 'router'): (output_inv_srcs, output_inv_dsts),
            ('channel', 'input', 'router'): (input_srcs, input_dsts),
            ('router', 'input_inv', 'channel'): (input_inv_srcs, input_inv_dsts),
        })

        return graph, pkt2id, rt2id, chan2id

    def _gen_router_attr(self, G, rt2id):
        """Generate router's attribute.
        Return: [Tensor(#Router, router_attr_dim)]

        Node attribute contains:
        - op_type: one-hot representation, dim=4
        """

        num_routers =  len(rt2id)
        op_type = torch.zeros(num_routers, 4)

        cast_op_type = {
            "wsrc_in": torch.tensor([1, 0, 0, 0, 0, 0]).unsqueeze(0),
            "insrc_in": torch.tensor([0, 1, 0, 0, 0, 0]).unsqueeze(0),
            "worker_in": torch.tensor([0, 0, 1, 0, 0, 0]).unsqueeze(0),
            "worker_out": torch.tensor([0, 0, 0, 1, 0, 0]).unsqueeze(0),
            "sink_out": torch.tensor([0, 0, 0, 0, 1, 0]).unsqueeze(0),
            "noc": torch.tensor([0, 0, 0, 0, 0, 1]).unsqueeze(0),
        }

        annotated = []
        for u, nattr in G.nodes(data=True):
            u_pe = nattr['p_pe']
            inject_port = self._get_port(-1, u_pe)
            if inject_port in rt2id.keys():
                op_type[inject_port:inject_port+1, :] += cast_op_type[nattr['op_type'] + "_in"]
                annotated.append(inject_port)
            eject_port = self._get_port(u_pe, -2)
            if eject_port in rt2id.keys():
                op_type[eject_port:eject_port+1, :] += cast_op_type[nattr['op_type'] + "_out"]
                annotated.append(eject_port)
        for port_id in rt2id.keys():
            if port_id not in annotated:
                op_type[port_id +1, :] += cast_op_type['noc']
        
        return {
            "op_type": op_type.float()
        }