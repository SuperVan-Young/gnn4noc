import pickle
import os
import networkx as nx

from trace_analyzer import TraceAnalyzer

dataset_root = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(dataset_root, "data")):
    os.mkdir(os.path.join(dataset_root, "data"))

class DGLFileGenerator:
    """Generate DGL files for training
    
    nattr: delay, op_type
    eattr: size, cnt, route
    labels: in_latency, out_latency (saved in nattr)
    
    """


    def __create_empty_array(self, array_size):
        """Create an empty core array for mapping
        """
        core_array = nx.MultiDiGraph()
        core_array.add_nodes_from(range(array_size**2))

        for u, nattr in core_array.nodes(data=True):
            core_array.nodes[u]["delay"] = 0
            core_array.nodes[u]["in_latency"] = 0
            core_array.nodes[u]["out_latency"] = 0
            core_array.nodes[u]["op_type"] = [0, 0, 0, 0] # wsrc, insrc, worker, sink

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
            # add edge information
            # We use max pid's information only
            if len(eattr["pkt"].keys()) == 0:
                continue
            pid = max(eattr["pkt"].keys())
            pkt_info = eattr["pkt"][pid]
            pkt_info["flit"] = eattr["size"]
            pkt_info["cnt"] = H.nodes[u]["cnt"]

            routing_hops = trace_analyzer.get_routing_hops(H.nodes[u]["p_pe"], H.nodes[v]["p_pe"], pid)
            for s, d in routing_hops:
                core_array.add_edge(s, d, pid, **pkt_info)

            # add delay information for computation nodes
            u_pe = H.nodes[u]["p_pe"]
            v_pe = H.nodes[v]["p_pe"]  # v doesn't mean virtual here
            core_array.nodes[u_pe]["delay"] = H.nodes[u]["delay"]

            # add packet latency to worker nodes
            onehot_wsrc = [1, 0, 0, 0]
            onehot_insrc = [0, 1, 0, 0]
            onehot_worker = [0, 0, 1, 0]
            onehot_sink = [0, 0, 0, 1]

            in_type, out_type = H.nodes[u]["op_type"], H.nodes[v]["op_type"]
            if out_type == "worker":
                core_array.nodes[v_pe]["op_type"] = onehot_worker
                if in_type == "wsrc":
                    core_array.nodes[u_pe]["op_type"] = onehot_wsrc
                elif in_type == "insrc":
                    core_array.nodes[u_pe]["op_type"] = onehot_insrc
                else:
                    raise RuntimeError

                in_edge = eattr["pkt"][pid]
                core_array.nodes[v_pe]["in_latency"] = in_edge["end_cycle"] - in_edge["start_cycle"]
            elif out_type == "sink":
                core_array.nodes[v_pe]["op_type"] = onehot_sink
                if in_type != "worker":
                    raise RuntimeError
                core_array.nodes[u_pe]["op_type"] = onehot_worker

                out_edge = eattr["pkt"][pid]
                core_array.nodes[u_pe]["out_latency"] = out_edge["end_cycle"] - out_edge["start_cycle"]
            else:
                raise RuntimeError


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


    def dump_graph(self, core_array:nx.DiGraph, layer:str):
        """Dump dgl file to target directory.
        """
        save_path = os.path.join(dataset_root, "data", f"{layer}.gpickle")
        
        with open(save_path, "wb+") as f:
            pickle.dump(core_array, f)



        