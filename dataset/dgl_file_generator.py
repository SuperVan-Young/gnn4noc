import os
from trace import Trace
import networkx as nx
import dgl

from trace_analyzer import TraceAnalyzer

dataset_root = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(dataset_root, "data")):
    os.mkdir(os.path.join(dataset_root, "data"))

class DGLFileGenerator:
    """Generate DGL files for training"""

    def __init__(self, array_size):
        self.array_size = array_size

    def __create_empty_array(self):
        """Create an empty core array for mapping
        """
        core_array = nx.MultiDiGraph()
        core_array.add_nodes_from(range(self.array_size*self.array_size))

        for u, nattr in core_array.nodes(data=True):
            core_array.nodes[u]["delay"] = 0

        return core_array

        
    def map_pe_array(self, trace_analyzer:TraceAnalyzer, layer:str, batch=0):
        """ Dump a multi-graph of #layer from trace analyzer.
        Layer should be a string
        Return: MultiDigraph of core array
        """
        G = trace_analyzer.graph.get_graph()
        nodes = [u for u, nattr in G.nodes(data=True) if nattr["layer"] == layer
            and nattr["batch"] == batch]
        assert len(nodes) > 0
        H = G.subgraph(nodes)
        
        core_array = self.__create_empty_array()

        for u, v, eattr in H.edges(data=True):
            # We use max pid's information only
            if len(eattr["pkt"].keys()) == 0:
                continue
            pid = max(eattr["pkt"].keys())
            pkt_info = eattr["pkt"][pid]
            # add more information about flit and cnt
            pkt_info["flit"] = eattr["size"]
            pkt_info["cnt"] = H.nodes[u]["cnt"]

            # add delay information for computation nodes
            p_pe = H.nodes[u]["p_pe"]
            core_array.nodes[p_pe]["delay"] = H.nodes[u]["delay"]

            routing_hops = trace_analyzer.get_routing_hops()
            for s, d in routing_hops:
                core_array.edges[s, d, pid] = pkt_info

        return core_array


    def aggregate_information(self, core_array:nx.MultiDiGraph):
        """Aggregate multiple edges into a single one
        Return: Digraph of core array
        """

        def agg_func(edges):
            eattr = dict()
            eattr["size"] = sum([e["flit"] * e["cnt"] for e in edges])
            eattr["cnt"] = sum([ e["cnt"] for e in edges])
            eattr["route"] = len(edges)
            return eattr

        H = nx.DiGraph()
        H.add_nodes_from(core_array.nodes(data=True))

        for u, v, data in core_array.edges(data=True, keys=False):
            H[u][v] = agg_func(data)

        # use induced subgraph to simplify
        nodes = [n for n, d in H.degree() if d > 0]
        H = H.subgraph(nodes)

        return H


    def dump_graph(self, core_array:nx.DiGraph, layer:str, category:str):
        """Dump dgl file to target directory.
        Category could be train, val or test.
        """
        assert category in ["train", "val", "test"]
        
        g = dgl.from_networkx(core_array, node_attrs=["delay"], edge_attrs=["size", "cnt", "route"])
        save_path = os.path.join(dataset_root, "data", category, f"{layer}.dgl")
        
        dgl.save_graphs(save_path)



        