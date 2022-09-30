import os
import sys
sys.path.append("..")

import networkx as nx
import torch
import dgl

from base import GraphGenerator, SmartDict, binarize_float

class HyperGraphGenerator(GraphGenerator):

    def _gen_graph(self, G:nx.DiGraph):
        graph, pkt2id, rt2id, chan2id = self._gen_skeleton(G)
        for attr, t in self._gen_packet_attr(G, pkt2id).items():
            graph.nodes['packet'].data[attr] = t
        for attr, t in self._gen_router_attr(G, rt2id).items():
            graph.nodes['router'].data[attr] = t
        for attr, t in self._gen_channel_attr(G, chan2id).items():
            graph.nodes['channel'].data[attr] = t

        return graph

    def _gen_skeleton(self, G:nx.DiGraph):
        """Generate heterograph given the subgraph.
        Returns:
        - graph: dgl heterogeneous graph
        - pkt2id: {first packet pid: packet id in graph}
        - rt2id: {p_pe: router id in graph}
        - chan2id: {(u_pe, v_pe): channel id in graph}
        """
        packet_to_channel = dict()  # pid: cids
        router_to_channel_in = dict()  # rid: cids
        router_to_channel_out = dict()  # rid: cids
        router_to_router = dict()  # rid: rids, direct connection

        pkt2id = SmartDict()
        rt2id = SmartDict()
        chan2id = SmartDict()

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != "data" or len(eattr["pkt"]) == 0:
                continue  # could happen when insrc and worker on the same pe

            pkt = eattr["pkt"][0]
            u_pe, v_pe = G.nodes[u]['p_pe'], G.nodes[v]['p_pe']
            channels = self.parser.routing_parser.get_routing_hops(u_pe, v_pe, pkt)

            pid = pkt2id[pkt]
            packet_to_channel[pid] = []
            for s, d in channels:
                rid_s, rid_d = rt2id[s], rt2id[d]
                cid = chan2id[(s, d)]

                packet_to_channel[pid].append(cid)        # packet pass this channel
                if rid_s not in router_to_channel_out.keys():
                    router_to_channel_out[rid_s] = []
                if cid not in router_to_channel_out[rid_s]:
                    router_to_channel_out[rid_s].append(cid)  # this channel is source router's output channel
                if rid_d not in router_to_channel_in.keys():
                    router_to_channel_in[rid_d] = []
                if cid not in router_to_channel_in[rid_d]:
                    router_to_channel_in[rid_d].append(cid)   # this channel is destination router's input channel

                # add direct connection between routers
                if rid_s not in router_to_router.keys():
                    router_to_router[rid_s] = []
                if rid_d not in router_to_router[rid_s]:
                    router_to_router[rid_s].append(rid_d)
                if rid_d not in router_to_router.keys():
                    router_to_router[rid_d] = []
                if rid_s not in router_to_router[rid_d]:
                    router_to_router[rid_d].append(rid_s)

        pass_srcs, pass_dsts = self._parse_edges_dict(packet_to_channel)
        pass_inv_srcs, pass_inv_dsts = pass_dsts.clone().detach(), pass_srcs.clone().detach()
        output_srcs, output_dsts = self._parse_edges_dict(router_to_channel_out)
        output_inv_srcs, output_inv_dsts = output_dsts.clone().detach(), output_srcs.clone().detach()
        input_dsts, input_srcs = self._parse_edges_dict(router_to_channel_in)
        input_inv_srcs, input_inv_dsts = input_dsts.clone().detach(), input_srcs.clone().detach()
        connect_srcs, connect_dsts = self._parse_edges_dict(router_to_router)


        graph = dgl.heterograph({
            ('packet', 'pass', 'channel'): (pass_srcs, pass_dsts),
            ('channel', 'pass_inv', 'packet'): (pass_inv_srcs, pass_inv_dsts),
            ('router', 'output', 'channel'): (output_srcs, output_dsts),
            ('channel', 'output_inv', 'router'): (output_inv_srcs, output_inv_dsts),
            ('channel', 'input', 'router'): (input_srcs, input_dsts),
            ('router', 'input_inv', 'channel'): (input_inv_srcs, input_inv_dsts),
            ('router', 'connect', 'router'): (connect_srcs, connect_dsts),
        })

        return graph, pkt2id, rt2id, chan2id

    def _parse_edges_dict(self, edges):
        """ Parse dict of edge lists.
        Return: source node tensor, dest node tensor
        """
        src, dst = [], []
        for s, l in edges.items():
            for d in l:
                src.append(s)
                dst.append(d)

        return torch.tensor(src), torch.tensor(dst)

    def _gen_router_attr(self, G, rt2id):
        """Generate router's attribute.
        Return: [Tensor(#Router, router_attr_dim)]

        Node attribute contains:
        - op_type: one-hot representation, dim=6
        """

        num_routers =  len(rt2id)
        op_type = torch.zeros(num_routers, 6)

        cast_op_type = {
            "wsrc": torch.tensor([1, 0, 0, 0, 0, 0]).unsqueeze(0),
            "insrc": torch.tensor([0, 1, 0, 0, 0, 0]).unsqueeze(0),
            "worker": torch.tensor([0, 0, 1, 0, 0, 0]).unsqueeze(0),
            "sink": torch.tensor([0, 0, 0, 1, 0, 0]).unsqueeze(0),
            "noc": torch.tensor([0, 0, 0, 0, 1, 0]).unsqueeze(0),
        }

        annotated = []
        for u, nattr in G.nodes(data=True):
            u_pe = nattr['p_pe']
            rid_u = rt2id[u_pe]
            op_type[rid_u:rid_u+1, :] += cast_op_type[nattr['op_type']]
            annotated.append(rid_u)
        for rid in rt2id.keys():
            if rid not in annotated:
                op_type[rid:rid+1, :] += cast_op_type['noc']
        
        inp = op_type.clone().detach()
        
        return {
            "op_type": op_type.float(),
            "inp": inp.float()
        }


    def _gen_packet_attr(self, G, pkt2id):
        """Generate packet's attribute.
        Return: [Tensor(#Packet, packet_attr_dim)

        Node attribute contains:
        - freq: frequency to send packet, dim=1
        - flit: size of flit (binarized float) dim=32
        """
        num_packets = len(pkt2id)
        freq = torch.zeros(num_packets)
        flit = torch.zeros(num_packets)
        delay = torch.zeros(num_packets)

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != "data" or len(eattr["pkt"]) == 0:
                continue  # could happen when insrc and worker on the same pe
            pkt = eattr["pkt"][0]  # the first pkt is fine
            pid = pkt2id[pkt]
            flit[pid:pid+1] = eattr['size']
            freq[pid:pid+1] = 1 / G.nodes[u]['delay']
            delay[pid:pid+1] = G.nodes[u]['delay']

        flit = binarize_float(flit, 32)
        freq = freq.unsqueeze(-1)
        delay = binarize_float(delay, 32)

        inp = torch.cat((flit, delay), dim=1)

        return {
            "flit": flit.float(),
            "freq": freq.float(),
            "delay": delay.float(),
            "inp": inp.float(),
        }

    
    def _gen_channel_attr(self, G, chan2id):
        """Generate channel's attribute.
        Return: [Tensor(#channel, channel_attr_dim)]

        Node attribute contains:
        - bandwidth: bytes per cycle, dim=32
        """
        num_channels = len(chan2id)
        bandwidth = torch.ones(num_channels).unsqueeze(-1)

        # this is a default bandwidth.
        # useless on current toolchain, but we leave this interface anyway.
        bandwidth = bandwidth * 1024  
        bandwidth = binarize_float(bandwidth, 32)

        inp = bandwidth.clone().detach()

        return {
            "bandwidth": bandwidth.float(),
            "inp": inp.float(),
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
