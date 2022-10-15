import os
import sys
import numpy as np
from tqdm import tqdm
sys.path.append("..")

import global_control as gc
from focus_agent.focus_agent import FocusAgent
from focus_agent.sampler import LayerSampler
from trace_parser.trace_parser import TraceParser

class ExperiencePredictor():
    def __init__(self, trace_parser:TraceParser) -> None:
        self.trace_parser = trace_parser

    def estimate_src_to_worker(self, layer_name=None):
        G = self.trace_parser.graph_parser.get_graph(layer=layer_name, batch=0)
        
        intervals = {
            'wsrc': 0,
            'insrc': 0,
        }
        is_multicast = {
            'wsrc': False,
            'insrc': False,
        }  # multicast edges collect with average, unicast edges collect with sum
        packets = {
            'wsrc': [],
            'insrc': [],
        }  # take each flow's first packet as a representative
        pid2hop = dict()
        pid2flit = dict()

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != 'data'or len(eattr['pkt']) == 0:
                continue
            src = G.nodes[u]['op_type']
            if src != 'wsrc' and src != 'insrc':
                continue
            
            # collect src information
            pid = eattr['pkt'][0]
            packets[src].append(pid)
            is_multicast[src] = is_multicast[src] or self.trace_parser.routing_parser.is_multicast_packet(pid)

            # collect edge information
            u_pe, v_pe = G.nodes[u]['p_pe'], G.nodes[v]['p_pe']
            pid2hop[pid] = len(self.trace_parser.routing_parser.get_routing_hops(u_pe, v_pe, pid))
            pid2flit[pid] = eattr['size']

        # estimate minimum src to worker interval
        for src in ['wsrc', 'insrc']:
            avg_buf = np.average([2 * pid2hop[pid] for pid in packets[src]])
            avg_flit = np.average([pid2flit[pid] for pid in packets[src]])
            intervals[src] = 2 * avg_flit + min(avg_flit / 2, avg_buf)
            if is_multicast[src]:
                pass
            else:
                intervals[src] *= len(packets[src])

        return intervals

    def estimate_worker_to_sink(self, layer_name=None):
        G = self.trace_parser.graph_parser.get_graph(layer=layer_name, batch=0)

        intervals = {n: 0 for n, op_type in G.nodes(data='op_type') if op_type == 'worker'}
        packets =  {n: -1 for n, op_type in G.nodes(data='op_type') if op_type == 'worker'}
        pid2hop = dict()
        pid2flit = dict()

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != 'data'or len(eattr['pkt']) == 0:
                continue
            src = G.nodes[u]['op_type']
            if src != 'worker':
                continue
            
            # collect src information
            pid = eattr['pkt'][0]
            packets[u] = pid

            # collect edge information
            u_pe, v_pe = G.nodes[u]['p_pe'], G.nodes[v]['p_pe']
            pid2hop[pid] = len(self.trace_parser.routing_parser.get_routing_hops(u_pe, v_pe, pid))
            pid2flit[pid] = eattr['size']

        # estimate minimum worker to sink interval
        for u in intervals.keys():
            pid = packets[u]
            buf = 2 * pid2hop[pid]
            flit = pid2flit[pid]
            intervals[u] = 2 * flit + min(flit / 2, buf)

        return intervals

    def predict_latency(self, layer_name=None):
        G = self.trace_parser.graph_parser.get_graph(layer=layer_name, batch=0)

        wsrc = [u for u, attr in G.nodes(data=True) if attr['op_type'] == 'wsrc']
        assert(len(wsrc) == 1)
        wsrc = wsrc[0]
        insrc = [u for u, attr in G.nodes(data=True) if attr['op_type'] == 'insrc']
        assert(len(insrc) == 1)
        insrc = insrc[0]
        workers = [u for u, attr in G.nodes(data=True) if attr['op_type'] == 'worker']
        sink = [u for u, attr in G.nodes(data=True) if attr['op_type'] == 'sink']
        assert(len(sink) == 1)
        sink = sink[0]

        # we consider a packet group's interval
        # e.g., 1 weight + 3 input -> 3 output, then a packet group is 1 weight packet + 3 input packet + 3 output packet
        group_size = {
            'wsrc': G.nodes[wsrc]['cnt'],
            'insrc': G.nodes[insrc]['cnt'],
            'worker': G.nodes[workers[0]]['cnt'],
        }
        min_group_size = min(group_size.values())
        group_size = {k: v / min_group_size for k, v in group_size.items()}

        # actual eject interval between two groups of src packets
        # is limited by both NoC throughput and src packets injection interval
        src_worker_throughput = self.estimate_src_to_worker(layer_name)
        wsrc_packet_recv_interval = max(src_worker_throughput['wsrc'], G.nodes[wsrc]['delay'])
        insrc_packet_recv_interval = max(src_worker_throughput['insrc'], G.nodes[insrc]['delay'])
        src_group_recv_interval = max(wsrc_packet_recv_interval * group_size['wsrc'], insrc_packet_recv_interval * group_size['insrc'])

        # actual injection interval between two groups of output packets
        # is limited by both recv interval and worker cores' computation capacity
        output_group_inject_interval = max(src_group_recv_interval, G.nodes[workers[0]]['delay'] * group_size['worker'])

        # actual eject interval between two groups of output packets
        # is limited by both NoC throughput and worker inject interval
        worker_intervals = self.estimate_worker_to_sink(layer_name)
        output_group_recv_interval = max(max(worker_intervals.values()) * group_size['worker'], output_group_inject_interval)

        # eventually, estimate layer's total latency
        group_cnt = min(G.nodes[wsrc]['cnt'], G.nodes[insrc]['cnt'], G.nodes[workers[0]]['cnt'])
        estimated_latency = group_cnt * output_group_recv_interval

        return estimated_latency

if __name__ == '__main__':
    agent = FocusAgent(fake_trace=True, simulate=True)

    tasknames = next(os.walk(gc.tasks_root))[1]
    pbar = tqdm(tasknames, desc="predict latency")

    latencies = []  # (estimation, ground truth)

    for taskname in pbar:
        root = os.path.join(gc.tasks_root, taskname)

        try:
            files = ['op_graph.gpickle', 'out.log', 'routing_board', 'spatial_spec']
            for file in files:
                assert os.path.exists(os.path.join(root, file))
        except:
            print(f"Build: missing data when converting {taskname}.")
            os.system(f"rm -r {root}")
            continue
        
        pbar.postfix = taskname

        graph_path = os.path.join(root, 'op_graph.gpickle')
        outlog_path = os.path.join(root, 'out.log')
        routing_path = os.path.join(root, 'routing_board')
        spec_path = os.path.join(root, 'spatial_spec')
        trace_parser = TraceParser(graph_path, outlog_path, routing_path, spec_path)

        predictor = ExperiencePredictor(trace_parser)
        estimated_latency = predictor.predict_latency()
        try:
            ground_truth = trace_parser.outlog_parser.get_total_latency()
        except:
            # timeout, but didn't delete this file
            continue

        pbar.set_description(f"{(estimated_latency - ground_truth) / ground_truth}")
        latencies.append((estimated_latency, ground_truth))

    errors = np.abs([(x - y) / y for x, y in latencies])
    print(f"mean = {np.average(errors)}")
    print(f"std = {np.std(errors)}")