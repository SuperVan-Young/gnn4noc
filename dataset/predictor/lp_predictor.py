import os
import sys
from matplotlib.pyplot import autoscale
import numpy as np
from tqdm import tqdm
from scipy.optimize import linprog
import networkx as nx
sys.path.append("..")

import global_control as gc
from focus_agent.focus_agent import FocusAgent
from focus_agent.sampler import LayerSampler
from trace_parser.trace_parser import TraceParser

class LinearProgrammingPredictor():
    """Linear Programming predictor for single layer.
    """

    def __init__(self, trace_parser:TraceParser) -> None:
        self.trace_parser = trace_parser
        self.G = None
        self.flow_cnt = 1  # remember to save for a fake variable
        self.A_ub = []  # list of 1d np.array
        self.b_ub = []  # list of 0d np.array
        self.A_eq = []  # list of 1d np.array
        self.b_eq = []  # list of 0d np.array

    def _get_port(self, s_pe, d_pe):
        directions = {
            (0, 1): 'north',
            (1, 0): 'east',
            (0, -1): 'south',
            (-1, 0): 'west',
        }
        if d_pe == -2:
            return 'eject'
        assert s_pe >= 0
        assert d_pe >= 0
        k = self.trace_parser.spec_parser.get_array_size()
        sx, sy = s_pe // k, s_pe % k
        dx, dy = d_pe // k, d_pe % k
        return directions[(dx-sx, dy-sy)]

    def _get_credit_stall_factor(self, flit):
        # on the long run, pipeline is withheld by credit
        factor = max(1, 4 / self.trace_parser.spec_parser.get_vc_buf_size())
        # but for short packets, virtual channels relieve this problem
        factor = min(factor, flit / 2) # the best dividend is 2
        return factor


    def _mark_computation_edges(self):
        """Mark flow from same computation source with the same flow id.
        (To distinguish with original fid, we use lpid instead)
        """
        G = self.G
        for u in G.nodes():
            has_flow = 0
            for _, v, eattr in G.out_edges(u, data=True):
                if eattr['edge_type'] != 'data'or len(eattr['pkt']) == 0:
                    G.edges[u, v]['lpid'] = -1  # will raise KeyError
                    continue
                G.edges[u, v]['lpid'] = self.flow_cnt
                has_flow = 1
            self.flow_cnt += has_flow  

    def _add_router_constraint(self):
        """Routers should transmit no more than 1 flit per cycle.
        """
        G = self.G
        
        k = self.trace_parser.spec_parser.get_array_size()
        routers = [{
            "north": [],  # lpid, cnt, delay, flit
            "south": [],
            "east": [],
            "west": [],
            "eject": [],
        } for _ in range(k ** 2)]
        finished_pkts = set()

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != 'data' or len(eattr['pkt']) == 0:
                continue

            flow_info = {
                'lpid': eattr['lpid'],
                'flit': eattr['size'],
                'cnt': G.nodes[u]['cnt'],
                'delay': G.nodes[u]['delay'],
            }
            
            # build a routing tree from hops
            u_pe, v_pe = G.nodes[u]['p_pe'], G.nodes[v]['p_pe']
            pid = eattr['pkt'][0]
            if pid in finished_pkts: continue  # debug: multicast pkts only count once
            finished_pkts.add(pid)

            hops = self.trace_parser.routing_parser.get_routing_hops(u_pe, v_pe, pid)
            routing_tree = {s: [] for s, d in hops}
            for s, d in hops:
                routing_tree[s].append(d)
            for s, d in hops:
                # add ejection channels for leaf routers
                if d not in routing_tree.keys():
                    routing_tree[d] = [-2]  # magic number

            for s_pe, ds in routing_tree.items():
                for d_pe in ds:
                    direction = self._get_port(s_pe, d_pe)
                    routers[s_pe][direction].append(flow_info)

        # constraints on all output ports
        # packets from the same flow cannot fully pipeline
        # pipeline_stall_factor = 3  # RC, VA, ..., W
        pipeline_stall_factor = 2  # The best I got is this

        for out_ports in routers:
            for direction, flows in out_ports.items():
                if len(flows) == 0: continue
                A_ub = np.zeros(self.flow_cnt)
                for flow in flows:
                    credit_stall_factor = self._get_credit_stall_factor(flow['flit'])
                    A_ub[flow['lpid']] += (flow['flit'] + pipeline_stall_factor) * credit_stall_factor
                self.A_ub.append(A_ub) 
                self.b_ub.append(np.ones(1))

    def _add_dependency_constraints(self):
        """Output flow's injection rate is limited by input flow
        """
        G = self.G

        for u, attr in G.nodes(data=True):
            if attr['op_type'] != 'worker': continue
            for _, _, eattr_out in G.out_edges(u, data=True):
                if eattr_out['lpid'] == -1: continue
                for src, _, eattr_in in G.in_edges(u, data=True):
                    if eattr_in['lpid'] == -1: continue
                    A_ub = np.zeros(self.flow_cnt)
                    A_ub[eattr_out['lpid']] = 1
                    A_ub[eattr_in['lpid']] = - G.nodes[src]['delay'] / G.nodes[u]['delay']
                    self.A_ub.append(A_ub)
                    self.b_ub.append(np.zeros(1))

    def _add_source_constraints(self):
        """Flow's injection rate is no more than production rate
        """
        G = self.G
        
        for u, _, eattr in G.edges(data=True):
            if eattr['lpid'] == -1: continue
            A_ub = np.zeros(self.flow_cnt)
            A_ub[eattr['lpid']] = G.nodes[u]['delay']
            self.A_ub.append(A_ub)
            self.b_ub.append(np.ones(1))

    def _add_queueing_constraints(self):
        """Previous packet blocks later packet in the same flow.
        """
        G = self.G
        # add interval in the same flow
        lpid_to_interval = {i: 0 for i in range(1, self.flow_cnt)}

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != 'data' or len(eattr['pkt']) == 0:
                continue
            lpid = eattr['lpid']

            u_pe, v_pe = G.nodes[u]['p_pe'], G.nodes[v]['p_pe']
            pid = eattr['pkt'][0]
            num_hops = len(self.trace_parser.routing_parser.get_routing_hops(u_pe, v_pe, pid))
            
            # solve a simple chasing problem, and get this...
            credit_stall_factor = self._get_credit_stall_factor(eattr['size'])
            packet_interval = min(num_hops, eattr['size']) * (1 - credit_stall_factor / 4)
            if self.trace_parser.routing_parser.is_multicast_packet(pid):
                lpid_to_interval[lpid] = packet_interval + eattr['size'] * credit_stall_factor
            else:
                lpid_to_interval[lpid] += packet_interval + eattr['size'] * credit_stall_factor

        for lpid, interval in lpid_to_interval.items():
            A_ub = np.zeros(self.flow_cnt)
            A_ub[lpid] = 1
            self.A_ub.append(A_ub)
            self.b_ub.append(np.ones(1) / interval)

    def _add_optimization_constraint(self):
        """Choose the slowest output flow as optimization target
        """
        G = self.G

        for u, v, eattr in G.edges(data=True):
            if G.nodes[v]['op_type'] != 'sink': continue
            if eattr['lpid'] == -1: continue
            A_ub = np.zeros(self.flow_cnt)
            A_ub[0] = 1
            A_ub[eattr['lpid']] = -1
            self.A_ub.append(A_ub)
            self.b_ub.append(np.zeros(1))

    def run(self, layer_name=None):
        # reset parameters
        self.G = self.trace_parser.graph_parser.get_graph(layer_name, batch=0)
        self.flow_cnt = 1
        self.A_ub = []
        self.b_ub = []
        self.A_eq = []
        self.b_eq = []

        self._mark_computation_edges()
        self._add_router_constraint()
        self._add_dependency_constraints()
        self._add_source_constraints()
        self._add_queueing_constraints()
        self._add_optimization_constraint()

        c = np.zeros(self.flow_cnt)
        c[0] = -1
        A_ub = np.stack(self.A_ub)
        b_ub = np.stack(self.b_ub)
        bounds = [(0, None) for _ in range(self.flow_cnt)]
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

        # print(A_ub)
        # print(b_ub)
        # print(result)

        speed = result.x[0]
        worker = [n for n, attr in self.G.nodes(data=True) if attr['op_type'] == 'worker'][0]
        cnt_worker = self.G.nodes[worker]['cnt']
        predicted_latency = cnt_worker / speed

        return predicted_latency

def test_fake_layers():
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

        predictor = LinearProgrammingPredictor(trace_parser)
        estimated_latency = predictor.run()
        try:
            ground_truth = trace_parser.outlog_parser.get_total_latency()
        except:
            # timeout, but didn't delete this file
            continue

        pbar.set_description(f"{(estimated_latency - ground_truth) / ground_truth}")
        latencies.append((estimated_latency, ground_truth))

    errors = np.array([(x - y) / y for x, y in latencies])
    print(f"mean = {np.average(errors)}")
    print(f"std = {np.std(errors)}")

    return errors

def test_example():
    agent = FocusAgent(fake_trace=True, simulate=True)
    taskname ="cw64_ci64_co64_bw0_bi1_fw2_fi7_fo12_dw1_di1_do1_n8" + f"_b1w1024_8x8"
    trace_parser = TraceParser(
        agent.get_op_graph_path(taskname),
        agent.get_outlog_path(taskname),
        agent.get_routing_path(taskname),
        agent.get_spec_path(taskname),
    )

    predictor = LinearProgrammingPredictor(trace_parser)
    result = predictor.run()
    print(result)

if __name__ == '__main__':
    # test_fake_layers()
    test_example()
