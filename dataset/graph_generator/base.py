import os
import sys
sys.path.append("..")

import pickle as pkl
import dgl
import numpy as np
import torch
from scipy import stats
from collections import UserDict

from trace_parser.trace_parser import TraceParser

class SmartDict(UserDict):
    """Automatically record new key's ID incrementally from 0
    """
    def __init__(self):
        super().__init__()
        self.__cnt = 0

    def __getitem__(self, x):
        if x not in self.data.keys():
            self.data[x] = self.__cnt
            self.__cnt += 1
        return self.data[x]

    def __len__(self):
        return self.__cnt


class GraphGenerator():
    """Generating graph from parser."""

    def __init__(self, parser: TraceParser, predict: bool) -> None:
        self.parser = parser
        self.predict = predict

    def generate_graph(self, layer=None, batch=None):
        """Generate subgraph if layer and batch are given, else the whole graph.
        """
        graph = self.__gen_graph(layer, batch)
        return graph

    def generate_label(self, layer=None, batch=None):
        label = self.__gen_label(layer, batch)
        return label

    def save_data(self, save_path, graph, label=None):
        saving = graph if self.predict else (graph, label)
        with open(save_path, "wb") as f:
            pkl.dump(saving, f)

    def __gen_graph(self, layer=None, batch=None):
        G = self.parser.graph_parser.get_graph(layer, batch)

        graph, pkt2id, rt2id = self.__gen_hetero_graph(G)
        for attr, t in self.__gen_hyper_nattr(G, pkt2id).items():
            graph.nodes['packet'].data[attr] = t
        for attr, t in self.__gen_nattr(G, rt2id).items():
            graph.nodes['router'].data[attr] = t

        return graph

    def __gen_hetero_graph(self, G):
        """Generate heterograph given the subgraph.
        Return: dgl.heterograph, pkt2id, rt2id
        """
        packet_to_router = dict()  # pid: rids
        router_to_router = dict()  # rid: rids

        pkt2id = SmartDict()
        rt2id = SmartDict()

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != "data":
                continue
            pkt = list(eattr["pkt"].keys())[0]  # the first pkt is fine
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
        - frequency to send packet, dim=1
        - flit, dim=32
        """
        num_packets = len(pkt2id)
        freq = torch.zeros(num_packets, 1)
        flit = torch.zeros(num_packets, 1)

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != "data":
                continue
            pkt = list(eattr["pkt"].keys())[0]  # the first pkt is fine
            pid = pkt2id[pkt]
            flit[pid:pid+1, :] = eattr['size']
            freq[pid:pid+1, :] = 1 / G.nodes[u]['delay']

        flit = self.__binarize_float(flit, 32)

        return {
            "flit": flit.float(),
            "freq": freq.float(),
        }

    def __binarize_float(self, tensor, bit):
        """Binarize float into integer, represented in vector
        Input: tensor(N, 1)
        Return: tensor(N, bit)
        """
        
        assert len(tensor.shape) == 2
        assert tensor.shape[1] == 1
        N, _ = tensor.shape
        bin_tensor = torch.zeros(N, bit)

        for i in range(bit):
            bin_tensor[:, i:i+1] = tensor % (2 ** (i+1))
        return bin_tensor

    def __gen_label(self, G):
        """Calculate congestion ratio.
        Return: Tensor(1,)
        """

        wsrc = [n for n, attr in G.nodes(data=True) if attr["op_type"] == "wsrc"]
        assert len(wsrc) == 1
        wsrc = wsrc[0]
        insrc = [n for n, attr in G.nodes(data=True) if attr["op_type"] == "insrc"]
        assert len(insrc) == 1
        insrc = insrc[0]
        workers = [n for n, attr in G.nodes(data=True) if attr["op_type"] == "worker"]
        assert len(workers) > 0
    
        w_cnt = int(G.nodes[wsrc]["cnt"])
        in_cnt = int(G.nodes[insrc]["cnt"])
        w_start = [float("inf")] * w_cnt
        w_end = [-float("inf")] * w_cnt
        in_start = [float("inf")] * in_cnt
        in_end = [-float("inf")] * in_cnt

        for w in workers:
            w_edges = G.edges[wsrc, w]["pkt"]
            w_pids = sorted(list(w_edges.keys()))

            in_edges = G.edges[insrc, w]["pkt"]
            in_pids = sorted(list(in_edges.keys()))

            for t in range(w_cnt):
                w_pkt = w_edges[w_pids[t]]
                w_start[t] = min(w_start[t], w_pkt["start_cycle"])
                w_end[t] = max(w_end[t], w_pkt["end_cycle"])

            for t in range(in_cnt):
                in_pkt = in_edges[in_pids[t]]
                in_start[t] = min(in_start[t], in_pkt["start_cycle"])
                in_end[t] = max(in_end[t], in_pkt["end_cycle"])

        # align different cnts
        if w_cnt > in_cnt:
            ratio = w_cnt // in_cnt
            w_latency = {i: l[1]-l[0] for i, l in enumerate(zip(w_start, w_end))}
            in_latency = {ratio*i: l[1]-l[0] for i, l in enumerate(zip(in_start, in_end))}
        elif w_cnt < in_cnt:
            ratio = in_cnt // w_cnt
            w_latency = {ratio*i: l[1]-l[0] for i, l in enumerate(zip(w_start, w_end))}
            in_latency = {i: l[1]-l[0] for i, l in enumerate(zip(in_start, in_end))}
        else:
            w_latency = {i: l[1]-l[0] for i, l in enumerate(zip(w_start, w_end))}
            in_latency = {i: l[1]-l[0] for i, l in enumerate(zip(in_start, in_end))}

        # Congestion is measured by a latency ratio.
        w_x = np.array(list(w_latency.keys()))
        w_y = np.array(list(w_latency.values()))
        in_x = np.array(list(in_latency.keys()))
        in_y = np.array(list(in_latency.values()))
        
        # TODO: We ignore cases when number of iteration is small
        # ignore the first irregular data due to instr. flow
        w_congestion = max(stats.linregress(w_x[1:], w_y[1:]).slope, 0) if len(w_x) > 4 else 0
        in_congestion = max(stats.linregress(in_x[1:], in_y[1:]).slope, 0) if len(in_x) > 4 else 0

        # adjust workload ratio
        workload = min(int(G.nodes[wsrc]['delay']), int(G.nodes[insrc]['delay']))
        w_congestion = w_congestion / workload
        in_congestion = in_congestion / workload

        # normalize
        congestion = max(w_congestion, in_congestion)
        congestion = torch.tensor(congestion)
        return congestion