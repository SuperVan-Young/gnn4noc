import os
import sys

import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from outlog_parser import OutlogParser
from graph_parser import GraphParser

class OpGraphVisualizer:

    def __init__(self, save_path, graph_path, outlog_path=None) -> None:
        """
        Visualize Computation graph in multipartite fashion.

        save_path: where to save the plotted graph
        graph_path: path of opgraph_*.gpickle
        outlog_path: path of out.log, if you need latency information on your graph
        """
        self.save_path = save_path
        self.graph_parser = GraphParser(graph_path)
        self.outlog_parser = OutlogParser(outlog_path) if outlog_path else None

    def plot(self):
        G = self.graph_parser.get_graph()

        # add virtual layer subset to nodes
        def extract_layer_number(s):
            s = re.search("layer\d+", s).group(0)
            s = re.search("\d+", s).group(0)
            return int(s)
        ot2vl = {
            "wsrc": 0,
            "insrc": 0,
            "worker": 1,
            "sink": 2,
        } # op_type -> virtual layer
        
        vlayer_count = dict()
        for n in G.nodes:
            node = G.nodes[n]
            node["vlayer"] = v = 3 * extract_layer_number(node["layer"]) + ot2vl[node["op_type"]]
            if v in list(vlayer_count.keys()):
                vlayer_count[v] += 1
            else:
                vlayer_count[v] = 1
        pos = nx.multipartite_layout(G, subset_key="vlayer", align="vertical")

        # define the figure based on compuation graph's scale
        scaling = 2 if max(vlayer_count.values()) >= 8 else 1  
        plt.figure(figsize=[min(len(vlayer_count) * scaling, 2**16), max(vlayer_count.values())])

        node_types = ["wsrc", "insrc", "worker", "sink"]
        edge_types = ["data", "control", "map_constraint"]
        
        # draw nodes
        node_color_map = {node_type: i / 4 for node_type, i in zip(node_types, range(4))}
        node_color = [node_color_map[node_type] for _, node_type in G.nodes(data="op_type")]
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap("Dark2"), node_color=node_color)

        # draw edges
        for edge_type in edge_types:
            edge_list, edge_color, edge_cmap = self._get_edge_color(edge_type)
            nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=edge_color, edge_cmap=edge_cmap)

        # draw node labels
        for node_type in edge_types:
            node_label = self._get_node_label(node_type)
            nx.draw_networkx_labels(G, pos, labels=node_label, font_size=8, alpha=0.8)

        # draw edge labels
        for edge_type in edge_types:
            edge_labels = self._get_edge_label(edge_type)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, alpha=0.5, font_size=8)

        # save the whole graph
        plt.savefig(self.save_path)
        plt.clf()

    def _get_node_label(self, node_type:str):
        """
        Returns:
        - node_label: {node: label}
        """
        G = self.graph_parser.get_graph()
        node_label = dict()

        # for example, label = cnt * delay
        if node_type == "insrc":
            node_label = {n: nattr['delay'] * nattr['cnt'] for n, nattr in G.nodes(data=True) if nattr['op_type'] == node_type}
        elif node_type == "wsrc":
            node_label = {n: nattr['delay'] * nattr['cnt'] for n, nattr in G.nodes(data=True) if nattr['op_type'] == node_type}
        elif node_type == "worker":
            node_label = {n: nattr['delay'] for n, nattr in G.nodes(data=True) if nattr['op_type'] == node_type}
        elif node_type == "sink":
            pass
        
        return node_label

    def _get_edge_label(self, edge_type:str):
        """
        Returns:
        - edge_label: {edge: label}
        """
        G = self.graph_parser.get_graph()
        edge_label = dict()

        if edge_type == "data":
            assert self.outlog_parser != None
            def get_last_packet_latency(u, v):
                u_pe, v_pe = G.nodes[u]['p_pe'], G.nodes[v]['p_pe']
                pkt = G.edges[u, v]["pkt"]
                if len(pkt) > 0:
                    lat = self.outlog_parser.get_latency(u_pe, v_pe, max(pkt))
                    lat = lat['end_cycle'] - lat['start_cycle']
                    lat = int(lat)
                else:
                    lat = 0
                return lat
            edge_label = {(u, v): get_last_packet_latency(u, v) for u, v, eattr in G.edges(data=True) if eattr['edge_type'] == edge_type}
        
        return edge_label
        

    def _get_edge_color(self, edge_type:str):
        """
        Returns:
        - edge_list : [(u, v)]
        - edge_color: str, or [edge_color_depth]
        - edge_cmap: matplotlib colormap if edge_color is list, else None
        """
        G = self.graph_parser.get_graph()
        edge_list = edge_list = [(u, v) for u, v, eattr in G.edges(data=True) if eattr["edge_type"] == edge_type]
        edge_color = None
        edge_cmap = None

        if edge_type == "data":
            # for example, color depth indicate a flow's last packet latency
            assert self.outlog_parser != None
            def get_last_packet_latency(u, v):
                u_pe, v_pe = G.nodes[u]['p_pe'], G.nodes[v]['p_pe']
                pkt = G.edges[u, v]["pkt"]
                if len(pkt) > 0:
                    lat = self.outlog_parser.get_latency(u_pe, v_pe, max(pkt))
                    lat = lat['end_cycle'] - lat['start_cycle']
                else:
                    lat = 0
                return lat
            edge_color = [get_last_packet_latency(u, v) for u, v in edge_list]

            edge_cmap = plt.cm.Reds

        elif edge_type == "control":
            edge_color = "blue"
            
        elif edge_type == "map_constraint":
            edge_color = "black"

        return edge_list, edge_color, edge_cmap

if __name__ == "__main__":
    save_path = "test.png"
    taskname = "bert-large_b1w1024_32x32"
    graph_path = "/home/xuechenhao/focus_scheduler/buffer/op_graph/op_graph_" + taskname + ".gpickle"
    outlog_path = "/home/xuechenhao/focus_scheduler/simulator/tasks/" + taskname + "/out.log"

    visualizer = OpGraphVisualizer(save_path, graph_path, outlog_path)
    visualizer.plot()
