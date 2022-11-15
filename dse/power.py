import os
import yaml
import re
import numpy as np

import dse_global_control as gc
from dataset.trace_parser.trace_parser import TraceParser

class PowerPredictor():
    """Predict a layer's power on a specific WSC architecture.
    """
    def __init__(self, **kwargs) -> None:
        self.task_root = kwargs['task_root']

        # FOCUS parameter
        self.core_array_size = kwargs['core_array_size']
        self.flit_size = kwargs['flit_size']

    def run(self):
        """Give a detailed report of arch power estimation.
        """
        result = dict()
        result['mac'] = self.calc_mac_power()
        result['noc'] = self.calc_noc_power()
        result['sram'] = self.calc_sram_power()
        result['reticle'] = self.calc_reticle_power()
        return result


    def calc_mac_power(self):
        model_pattern = re.compile(r"^(.*)_layer\d+$")
        dims = ['C', 'M', 'N', 'P', 'Q', 'R', 'S']
        report = dict()

        benchmark_root = os.path.join(self.task_root, 'benchmark')
        for _, __, files in os.walk(benchmark_root):
            for file in files:
                with open(os.path.join(benchmark_root, file), 'r') as f:
                    bm = yaml.load(f, Loader=yaml.FullLoader)
                for model_full_name, layers in bm.items():
                    for layer_name, cores in layers.items():
                        model_name = model_pattern.match(layer_name).group(1)
                        layer_spec_path = os.path.join(gc.database_root, model_name, f"{layer_name}.yaml")
                        assert os.path.exists(layer_spec_path)
                        with open(layer_spec_path, 'r') as f:
                            layer_spec = yaml.load(f, Loader=yaml.FullLoader)
                        mac_total = np.prod([v for k, v in layer_spec['problem']['instance'] if k in dims])
                        report[layer_name] = mac_total
            break
        
        return report


    def calc_noc_power(self):
        report = dict()

        benchmark_names = []
        benchmark_root = os.path.join(self.task_root, 'benchmark')
        for _, __, files in os.walk(benchmark_root):
            for file in files:
                with open(os.path.join(benchmark_root, file), 'r') as f:
                    bm = yaml.load(f, Loader=yaml.FullLoader)
                for model_full_name, layers in bm.items():
                    benchmark_names.append(benchmark_names)
                    
        for benchmark_name in benchmark_names:
            taskname = f"{benchmark_name}_b1w{self.flit_size}_{self.core_array_size}x{self.core_array_size}"
            graph_path = gc.get_op_graph_path(taskname)
            routing_path = gc.get_routing_path(taskname)
            spec_path = gc.get_spec_path(taskname)
            assert os.path.exists(graph_path), f"graph_path {graph_path} doesn't exist!"
            assert os.path.exists(routing_path), f"routing_path {routing_path} doesn't exist!"
            assert os.path.exists(spec_path), f"spec_path {spec_path} doesn't exist!"

            trace_parser = TraceParser(
                graph_path=graph_path,
                outlog_path=None,
                routing_path=routing_path,
                spec_path=spec_path
            )

            for layer_name in trace_parser.graph_parser.get_layers():
                G = trace_parser.get_graph(layer_name, batch=0)

                # copied from lp predictor
                pkt_infos = dict()
                for u, v, eattr in G.edges(data=True):
                    if eattr['edge_type'] != 'data' or len(eattr['pkt']) == 0:
                        continue

                    # build a routing tree from hops (add eject channels later)
                    u_pe, v_pe = G.nodes[u]['p_pe'], G.nodes[v]['p_pe']
                    pid = eattr['pkt'][0]
                    if pid in pkt_infos.keys():
                        # for multicast packets, add its ejection channel
                        pkt_infos[pid]['channels'] += 1
                        continue

                    hops = self.trace_parser.routing_parser.get_routing_hops(u_pe, v_pe, pid)
                    
                    pkt_infos[pid] = {
                        'channels': len(hops) + 2,  # 1 inject + 1 eject
                        'cnt': G.nodes[u]['cnt'],
                        'flit': eattr['size'],
                    }

                total_flits = np.sum([info['channels'] * info['cnt'] * info['flit'] for pid, info in pkt_infos.items()])
                report[layer_name] = total_flits


    def calc_sram_power(self):
        pass

    def calc_reticle_power(self):
        pass

