import os
import sys
from xml.dom.minicompat import NodeList
sys.path.append("..")

import networkx as nx
import torch
import dgl

from base import GraphGenerator, SmartDict, binarize_float

class HyperGraphGenerator(GraphGenerator):

    def _gen_graph(self, G:nx.DiGraph):
        graph, pkt2id, rt2id = self.__gen_skeleton(G)
        for attr, t in self.__gen_hyper_nattr(G, pkt2id).items():
            graph.nodes['packet'].data[attr] = t
        for attr, t in self.__gen_nattr(G, rt2id).items():
            graph.nodes['router'].data[attr] = t

        return graph

    def _gen_label(self, G:nx.DiGraph):
        pass

    def __gen_skeleton(self, G:nx.DiGraph):
        """Generate heterograph given the subgraph.
        Return: dgl.heterograph, pkt2id, rt2id
        """
        packet_to_router = dict()  # pid: rids
        router_to_router = dict()  # rid: rids

        pkt2id = SmartDict()  # FIRST packet: pid
        rt2id = SmartDict()

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != "data":
                continue
            pkt = eattr["pkt"][0]  # the first pkt is fine
            u_pe, v_pe = G.nodes[u]['p_pe'], G.nodes[v]['p_pe']
            routing = self.parser.routing_parser.get_routing_hops(u_pe, v_pe, pkt)

            pid = pkt2id[pkt]
            rid_u = rt2id[u_pe]
            packet_to_router[pid] = [rid_u]  # add routing start

            for s, e in routing:
                rid_s, rid_e = rt2id[s], rt2id[e]
                packet_to_router[pid].append(rid_e)  # add every routing hop's end

                # for every hop, add physical channel connection
                if rid_s not in router_to_router.keys():
                    router_to_router[rid_s] = []
                if rid_e not in router_to_router[rid_s]:
                    router_to_router[rid_s].append(rid_e)

        pass_edges = self.__parse_edges_dict(packet_to_router)
        transfer_edges = (pass_edges[1].clone().detach(), pass_edges[0].clone().detach())
        connect_edges = self.__parse_edges_dict(router_to_router)
        backpressure_edges = (connect_edges[1].clone().detach(), connect_edges[0].clone().detach())
        graph = dgl.heterograph({
            ('packet', 'pass', 'router'): pass_edges,
            ('router', 'transfer', 'packet'): transfer_edges,
            ('router', 'connect', 'router'): connect_edges,
            ('router', 'backpressure', 'router'): backpressure_edges,
        })

        return graph, pkt2id, rt2id

    def __parse_edges_dict(self, edges):
        """ Parse dict of edge lists.
        Return: source node tensor, dest node tensor
        """
        src, dst = [], []
        for s, l in edges.items():
            for d in l:
                src.append(s)
                dst.append(d)

        return torch.tensor(src), torch.tensor(dst)

    def __gen_nattr(self, G, rt2id):
        """Generate router's attribute.
        Return: [Tensor(#Router, node_attr_dim)]

        Node attribute contains:
        - op_type: one-hot representation, dim=4
        """

        num_routers =  len(rt2id)
        op_type = torch.zeros(num_routers, 4)

        cast_op_type = {
            "wsrc": torch.tensor([1, 0, 0, 0]).unsqueeze(0),
            "insrc": torch.tensor([0, 1, 0, 0]).unsqueeze(0),
            "worker": torch.tensor([0, 0, 1, 0]).unsqueeze(0),
            "sink": torch.tensor([0, 0, 0, 1]).unsqueeze(0),
        }

        for u, nattr in G.nodes(data=True):
            u_pe = nattr['p_pe']
            rid_u = rt2id[u_pe]
            op_type[rid_u:rid_u+1, :] = cast_op_type[nattr['op_type']]
        
        return {
            "op_type": op_type.float()
        }


    def __gen_hyper_nattr(self, G, pkt2id):
        """Generate packet's attribute.
        Return: [Tensor(#Packet, hyper_node_dim)

        Node attribute contains:
        - freq: frequency to send packet, dim=1
        - flit: size of flit (binarized float) dim=32
        """
        num_packets = len(pkt2id)
        freq = torch.zeros(num_packets)
        flit = torch.zeros(num_packets)

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != "data":
                continue
            pkt = eattr["pkt"][0]  # the first pkt is fine
            pid = pkt2id[pkt]
            flit[pid:pid+1] = eattr['size']
            freq[pid:pid+1] = 1 / G.nodes[u]['delay']

        flit = binarize_float(flit, 32)
        freq = freq.unsqueeze(-1)

        return {
            "flit": flit.float(),
            "freq": freq.float(),
        }

    def _gen_label(self, G: nx.DiGraph):
        """Calculate maximum congestion ratio.
        Use throughput to derive it.
        Return: Tensor(1,)
        """
        srcs = [n for n, op_type in G.nodes(data="op_type")
                if op_type == 'wsrc' or op_type == 'insrc']
        workers  = [n for n, op_type in G.nodes(data="op_type")
                if op_type == 'worker']

        node2pe = lambda x: G.nodes[x]["p_pe"]

        max_slope = 0  # throughput \mu = #cnt / #delay_total
                       # k = \lambda / \mu = #delay_total / (#cnt * #delay)
        for s in srcs:
            cnt = G.nodes[s]['cnt']
            delay = G.nodes[s]['delay']
            sid = node2pe(s)

            for w in workers:
                p = self.parser.outlog_parser
                wid = node2pe(w)
                pid2lat = lambda pid: p.get_latency(sid, wid, pid)["end_cycle"]

                end_cycles = [pid2lat(pid) for pid in G.edges[s, w]['pkt']]
                duration = max(end_cycles) - min(end_cycles)
                slope = duration / (cnt * delay)
                max_slope = max(max_slope, slope)

        return torch.tensor(max_slope)
