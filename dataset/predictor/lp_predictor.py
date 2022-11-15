import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.optimize import linprog
sys.path.append("..")

import global_control as gc
from focus_agent.focus_agent import FocusAgent
from focus_agent.sampler import LayerSampler
from trace_parser.trace_parser import TraceParser

class LinearProgrammingPredictor():
    """Linear Programming predictor for single layer.
    """

    def __init__(self, trace_parser:TraceParser, noc_spec=None) -> None:
        self.trace_parser = trace_parser
        self.noc_spec = noc_spec
        self.G = None
        self.flow_cnt = 1  # remember to save for a fake variable
        self.A_ub = []  # list of 1d np.array
        self.b_ub = []  # list of 0d np.array
        self.A_eq = []  # list of 1d np.array
        self.b_eq = []  # list of 0d np.array

    def _get_credit_stall_factor(self, flit):
        # on the long run, pipeline is withheld by credit
        factor = max(1, 4 / self.trace_parser.spec_parser.get_vc_buf_size())
        # but for very short packets, virtual channels relieve this problem
        thresh = 4
        if flit <= thresh:
            factor = (factor - 1) / (thresh - 2) * (flit - 2) + 1
        return factor

    def _get_pipeline_stall_factor(self):
        # need approximately 2 cycles to acquire vc
        return 2

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
        routers = [dict() for _ in range(k ** 2)]
        finished_pkts = set()
        weight_pkts = set()  # M spatial unroll on cores?

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != 'data' or len(eattr['pkt']) == 0:
                continue

            flow_types = {
                'wsrc': 'weight',
                'insrc': 'input',
                'worker': 'output',
            }

            flow_info = {
                'lpid': eattr['lpid'],
                'flit': eattr['size'],
                'cnt': G.nodes[u]['cnt'],
                'delay': G.nodes[u]['delay'],
                'flow_type': flow_types[G.nodes[u]['op_type']],
            }
            
            # build a routing tree from hops (add eject channels later)
            u_pe, v_pe = G.nodes[u]['p_pe'], G.nodes[v]['p_pe']
            pid = eattr['pkt'][0]
            if pid in finished_pkts:
                # for multicast packets, add its ejection channel
                routers[v_pe][-2].append(flow_info)
                continue
            finished_pkts.add(pid)
            if G.nodes[u]['op_type'] == 'wsrc': weight_pkts.add(pid)

            hops = self.trace_parser.routing_parser.get_routing_hops(u_pe, v_pe, pid)
            routing_tree = {s: [] for s, d in hops}
            for s, d in hops:
                routing_tree[s].append(d)
            routing_tree[v_pe].append(-2)

            for s_pe, ds in routing_tree.items():
                for d_pe in ds:
                    if d_pe not in routers[s_pe].keys():
                        routers[s_pe][d_pe] = []
                    routers[s_pe][d_pe].append(flow_info)

        # constraints on all output ports
        # packets from the same flow cannot fully pipeline
        pipeline_stall_factor = self._get_pipeline_stall_factor()

        for s_pe, out_ports in enumerate(routers):
            for d_pe, flows in out_ports.items():
                if len(flows) == 0: continue
                A_ub = np.zeros(self.flow_cnt)
                for flow in flows:
                    credit_stall_factor = self._get_credit_stall_factor(flow['flit'])
                    A_ub[flow['lpid']] += (flow['flit'] + pipeline_stall_factor) * credit_stall_factor
                    if flow['flow_type'] == 'output':
                        A_ub[flow['lpid']] /= len(weight_pkts)  # assume we have #M_DRAM_S sinks
                self.A_ub.append(A_ub)

                bw = self.noc_spec.get_relative_bandwidth(s_pe, d_pe) if self.noc_spec else 1
                self.b_ub.append(np.ones(1) * bw)

    def _add_dependency_constraints(self):
        """Output flow's injection rate is limited by input flow.
        Weight/input src input flow limits each other.
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
            
            # between flows, this is actually optional
            # wsrcs = [src for src, _ in G.in_edges(u) if G.nodes[src]['op_type'] == 'wsrc']
            # insrcs = [src for src, _ in G.in_edges(u) if G.nodes[src]['op_type'] == 'insrc']
            # assert(len(wsrcs) == 1)
            # assert(len(insrcs) == 1)
            # wsrc = wsrcs[0]
            # insrc = insrcs[0]
            # flow_w = G.edges[wsrc, u]['lpid']
            # flow_i = G.edges[insrc, u]['lpid']
            # A_eq = np.zeros(self.flow_cnt)
            # A_eq[flow_w] = G.nodes[wsrc]['delay']
            # A_eq[flow_i] = -G.nodes[insrc]['delay']
            # b_eq = np.zeros(1)
            # self.A_eq.append(A_eq)
            # self.b_eq.append(b_eq)

    def _add_source_constraints(self):
        """Flow's injection rate is no more than production rate
        """
        G = self.G
        
        for u, nattr in G.nodes(data=True):
            first_out_pkts = set()
            for _, _, eattr_out in G.out_edges(u, data=True):
                if eattr_out['lpid'] == -1: continue
                first_out_pkts.add(eattr_out['pkt'][0])
            for _, _, eattr_out in G.out_edges(u, data=True):
                if eattr_out['lpid'] == -1: continue
                A_ub = np.zeros(self.flow_cnt)
                A_ub[eattr_out['lpid']] = G.nodes[u]['delay'] + len(first_out_pkts)
                self.A_ub.append(A_ub)
                self.b_ub.append(np.ones(1))

    def _add_queueing_constraints(self):
        """Previous packet blocks later packet in the same flow.
        """
        G = self.G
        # add interval in the same flow
        lpid_to_interval = {i: 0 for i in range(1, self.flow_cnt)}
        lpid_pkts = {i: [] for i in range(1, self.flow_cnt)}

        k = self.trace_parser.spec_parser.get_array_size()
        def get_num_hops(s_pe, d_pe):
            sx, sy = s_pe // k, s_pe % k
            dx, dy = d_pe // k, d_pe % k
            return abs(sx-dx) + abs(sy-dy)

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != 'data' or len(eattr['pkt']) == 0:
                continue
            lpid = eattr['lpid']

            u_pe, v_pe = G.nodes[u]['p_pe'], G.nodes[v]['p_pe']
            pid = eattr['pkt'][0]
            num_hops = get_num_hops(u_pe, v_pe)
            
            # solve a simple chasing problem, and get this...
            credit_stall_factor = self._get_credit_stall_factor(eattr['size'])
            packet_interval = min(num_hops, eattr['size']) * (1 - credit_stall_factor / 4)
            lpid_to_interval[lpid] += packet_interval + eattr['size'] * credit_stall_factor
            lpid_pkts[lpid].append(pid)

        for lpid, interval in lpid_to_interval.items():
            pid = lpid_pkts[lpid][0]
            if self.trace_parser.routing_parser.is_multicast_packet(pid):
                interval /= len(lpid_pkts[lpid])
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
        A_eq = np.stack(self.A_eq) if self.A_eq != [] else None
        b_eq = np.stack(self.b_eq) if self.b_eq != [] else None
        bounds = [(0, None) for _ in range(self.flow_cnt)]
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        # print(A_ub)
        # print(b_ub)
        # print(result)

        speed = result.x[0]
        worker = [n for n, attr in self.G.nodes(data=True) if attr['op_type'] == 'worker'][0]
        cnt_worker = self.G.nodes[worker]['cnt']
        predicted_latency = cnt_worker / speed

        return predicted_latency

    def get_computation(self, layer_name=None):
        G = self.trace_parser.graph_parser.get_graph(layer_name, batch=0)
        
        latency = 0
        for u, attr in G.nodes(data=True):
            if attr['op_type'] == 'sink': continue
            latency = attr['delay'] * attr['cnt']
        return latency

    def get_data_transmission(self, layer_name=None):
        G = self.trace_parser.graph_parser.get_graph(layer_name, batch=0)
        srcs_info = {
            'wsrc': {
                'flit': 0,
                'unicast': 0,  # multicast pid only count once
                'cnt': 0,
                'total': 0,
            },
            "insrc": {
                'flit': 0,
                'unicast': 0,
                'cnt': 0,
                'total': 0,
            },
            "worker": {
                'flit': 0,
                'unicast': 0,
                'cnt': 0,
                'total': 0,
            },
        }  # wsrc/insrc: flit, cnt, total
        finished_pids = set()

        for u, v, eattr in G.edges(data=True):
            if eattr['edge_type'] != 'data' or len(eattr['pkt']) == 0:
                continue
            src_type = G.nodes[u]['op_type']
            if src_type != 'wsrc' and src_type != 'insrc' and src_type != 'worker': continue
            pid = eattr['pkt'][0]
            if pid in finished_pids: continue
            finished_pids.add(pid)
            srcs_info[src_type]['flit'] = eattr['size']
            srcs_info[src_type]['cnt'] = max(G.nodes[u]['cnt'], srcs_info[src_type]['cnt'])
            srcs_info[src_type]['unicast'] += 1
        
        for src_type in ['wsrc', 'insrc', 'worker']:
            srcs_info[src_type]['total'] = srcs_info[src_type]['flit'] * srcs_info[src_type]['unicast'] * srcs_info[src_type]['cnt']
        return srcs_info


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
