import os
import sys
import networkx as nx
sys.path.append("..")

import global_control as gc
from compiler.op_graph.micro_op_graph import MicroOpGraph

class GraphParser():
    def __init__(self, op_graph_path) -> None:
        self.op_graph_path = op_graph_path
        self.__op_graph = None

    def get_graph(self, layer=None, batch=None):
        if self.__op_graph == None:
            self.__parse_graph()
        G = self.__op_graph.get_graph()
        if layer != None and batch != None:
            nodes = [n for n, attr in G.nodes(data=True)
            if attr["layer"] == layer
            and attr["batch"] == batch]
            G = G.subgraph(nodes)
        return G

    def get_layers(self):
        if self.__op_graph == None:
            self.__parse_graph()
        layers = {nattr['layer'] for n, nattr in self.__op_graph.get_graph().nodes(data=True)}
        return layers

    def __parse_graph(self):
        assert os.path.exists(self.op_graph_path), f"op_graph {self.op_graph_path} not exists!"

        self.__op_graph = nx.read_gpickle(self.op_graph_path)

        # copied from toolchain's trace_generator
        # annotate pkt to each edge
        pkt_counter = 0

        op_graph = self.__op_graph.get_graph()

        assert nx.is_directed_acyclic_graph(op_graph)
        for _, __, eattr in op_graph.edges(data=True):
            eattr["pkt"] = []

        node2pe = lambda x: op_graph.nodes[x]["p_pe"]

        for node in nx.topological_sort(op_graph):

            nattr = op_graph.nodes[node]
            iteraction_cnt = int(nattr["cnt"])

            # propagate data to data edges
            out_data_edges = [(u, v) for u, v, t in op_graph.out_edges(node, data="edge_type") if t == "data" and node2pe(u) != node2pe(v)]
            for _ in range(iteraction_cnt):
                flows = {op_graph.edges[e]["fid"] for e in out_data_edges}
                fid_to_pid = {fid: pid for fid, pid in zip(flows, range(pkt_counter, pkt_counter + len(flows)))}
                pkt_counter += len(flows)

                for u, v in out_data_edges:
                    fid = op_graph.edges[u, v]["fid"]
                    pid = fid_to_pid[fid]
                    op_graph.edges[u, v]["pkt"].append(pid)

            # propagate control signals
            out_control_edges = [(u, v) for u, v, t in op_graph.out_edges(node, data="edge_type") if t == "control" and node2pe(u) != node2pe(v)]
            for u, v in out_control_edges:
                pid = pkt_counter
                pkt_counter += 1
                op_graph.edges[u, v]["pkt"].append(pid)