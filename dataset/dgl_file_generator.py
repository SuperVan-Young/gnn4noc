import pickle
import os
import numpy as np
from scipy import stats
import networkx as nx

from trace_analyzer import TraceAnalyzer

dataset_root = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(dataset_root, "data")):
    os.mkdir(os.path.join(dataset_root, "data"))


from collections import UserDict
class SmartDict(UserDict):
    """Automatically record new key's ID incrementally from 0
    """
    def __init__(self):
        super().__init__()
        self.cnt = 0

    def __getitem__(self, x):
        if x not in self.data.keys():
            self.data[x] = self.cnt
            self.cnt += 1
        return self.data[x]


class DGLFileGenerator:
    """Generate DGL files for training.
    """
    
    def dump_data(self, trace_analyzer: TraceAnalyzer, layer:str, batch=0):
        """Map packets of one tile to the Router array.
        Dataset could rebuild a HeteroGraph from these information.

        Return: {
            packet_to_router
            router_to_router
            cnt
            delay
            congestion
        }
        """

        # store routing information
        packet_to_router = dict()
        router_to_router = dict()

        pkt2id = SmartDict()
        rt2id = SmartDict()
        
        G = trace_analyzer.graph.get_graph()
        nodes = [n for n, attr in G.nodes(data=True)
            if attr["layer"] == layer
            and attr["batch"] == batch]
        H = G.subgraph(nodes)

        for u, v, eattr in H.edges(data=True):
            pkt = eattr["pkt"].keys()[0]
            routing = trace_analyzer.get_routing_hops(u, v, pkt)

            pid = pkt2id[pkt]
            packet_to_router[pid] = [rt2id[H.nodes[u]['p_pe']]]  # add routing start

            for s, e in routing:
                sid, eid = rt2id[s], rt2id[e]
                packet_to_router[pid] = eid  # add routing end of every hop

                # for every hop, add physical channel connection
                if sid not in router_to_router.keys():
                    router_to_router[sid] = []
                if eid not in router_to_router[sid]:
                    router_to_router[sid].append(eid)

        ########################################################################

        # store delay & cnt
        wsrc = [n for n, attr in G.nodes(data=True) if attr["op_type"] == "wsrc"]
        assert len(wsrc) == 1
        wsrc = wsrc[0]
        insrc = [n for n, attr in G.nodes(data=True) if attr["op_type"] == "insrc"]
        assert len(insrc) == 1
        insrc = insrc[0]
        workers = [n for n, attr in G.nodes(data=True) if attr["op_type"] == "worker"]
        assert len(workers) > 0

        cnt = {
            "wsrc": int(G.nodes[wsrc]["cnt"]),
            "insrc": int(G.nodes[insrc]["cnt"]),
            "worker": int(G.nodes[workers[0]]["cnt"]),
        }
        delay = {
            "wsrc": int(G.nodes[wsrc]["delay"]),
            "insrc": int(G.nodes[insrc]["delay"]),
            "worker": int(G.nodes[workers[0]]["delay"]),
        }  # actually you only need delay information

        ########################################################################
        
        # store congestion
        w_cnt = cnt['wsrc']
        in_cnt = cnt['insrc']
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
        
        # FIXME: We ignore cases when number of iteration is small
        # ignore the first irregular data due to instr. flow
        w_congestion = max(stats.linregress(w_x[1:], w_y[1:]).slope, 0) if len(w_x) > 4 else 0
        in_congestion = max(stats.linregress(in_x[1:], in_y[1:]).slope, 0) if len(in_x) > 4 else 0

        ########################################################################
        result = {
            "packet_to_router": packet_to_router,
            "router_to_router": router_to_router,
            "cnt": cnt,
            "delay": delay,
            "w_congestion": w_congestion,
            "in_congestion": in_congestion,
        }