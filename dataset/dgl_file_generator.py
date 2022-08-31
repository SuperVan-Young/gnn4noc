import pickle
import os
import networkx as nx

from trace_analyzer import TraceAnalyzer

dataset_root = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(dataset_root, "data")):
    os.mkdir(os.path.join(dataset_root, "data"))

class DGLFileGenerator:
    """Generate DGL files for training"""


    def __create_empty_array(self, array_size):
        """Create an empty core array for mapping
        """
        core_array = nx.MultiDiGraph()
        core_array.add_nodes_from(range(array_size**2))

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
        
        core_array = self.__create_empty_array(trace_analyzer.get_array_size())

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

            routing_hops = trace_analyzer.get_routing_hops(H.nodes[u]["p_pe"], H.nodes[v]["p_pe"], pid)
            for s, d in routing_hops:
                core_array.add_edge(s, d, pid, **pkt_info)

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

        agg_info = dict()
        for u, v, pid, data in core_array.edges(data=True, keys=True):
            if (u, v) not in agg_info.keys():
                agg_info[(u, v)] = []
            agg_info[(u, v)].append(data)
        
        for edge, info in agg_info.items():
            u, v = edge
            H.add_edge(u, v, **agg_func(info))

        # use induced subgraph to simplify
        nodes = [n for n, d in H.degree() if d > 0]
        H = H.subgraph(nodes).copy()

        return H


    def dump_graph(self, core_array:nx.DiGraph, layer:str, category:str):
        """Dump dgl file to target directory.
        Category could be train, val or test.
        """
        assert category in ["train", "val", "test"]
        save_path = os.path.join(dataset_root, "data", category, f"{layer}.gpickle")
        
        with open(save_path, "wb+") as f:
            pickle.dump(core_array, f)



        