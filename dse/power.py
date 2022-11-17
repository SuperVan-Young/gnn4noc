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

        # actual core array size
        self.core_array_h = kwargs['core_array_h']
        self.core_array_w = kwargs['core_array_w']

    def run(self, benchmark_path, full_benchmark_path):
        """Give a detailed report of arch power estimation.
        """
        assert os.path.exists(benchmark_path)
        assert os.path.exists(full_benchmark_path)

        result = dict()
        result['mac'] = self.calc_mac_power(benchmark_path)
        result['noc'] = self.calc_noc_power(benchmark_path)
        result['sram'] = self.calc_sram_power(benchmark_path)
        result['reticle'] = self.calc_reticle_power(full_benchmark_path)
        return result


    def calc_mac_power(self, benchmark_path):
        model_pattern = re.compile(r"^(.*)_layer\d+$")
        dims = ['C', 'M', 'N', 'P', 'Q', 'R', 'S']
        report = dict()

        with open(benchmark_path, 'r') as f:
            bm = yaml.load(f, Loader=yaml.FullLoader)
        for model_full_name, layers in bm.items():
            for layer in layers:
                for layer_name, cores in layer.items():
                    model_name = model_pattern.match(layer_name).group(1)
                    layer_spec_path = os.path.join(gc.database_root, model_name, f"{layer_name}.yaml")
                    assert os.path.exists(layer_spec_path)
                    with open(layer_spec_path, 'r') as f:
                        layer_spec = yaml.load(f, Loader=yaml.FullLoader)
                    mac_total = np.prod([v for k, v in layer_spec['problem']['instance'].items() if k in dims])
                    report[layer_name] = int(mac_total)
        
        return report


    def calc_noc_power(self, benchmark_path):
        report = dict()

        benchmark_names = []
        with open(benchmark_path, 'r') as f:
            bm = yaml.load(f, Loader=yaml.FullLoader)
        for model_full_name, layers in bm.items():
            benchmark_names.append(model_full_name)
                    
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
                G = trace_parser.graph_parser.get_graph(layer_name, batch=0)

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

                    hops = trace_parser.routing_parser.get_routing_hops(u_pe, v_pe, pid)
                    
                    pkt_infos[pid] = {
                        'channels': len(hops) + 2,  # 1 inject + 1 eject
                        'cnt': G.nodes[u]['cnt'],
                        'flit': eattr['size'],
                    }

                total_flits = np.sum([info['channels'] * info['cnt'] * info['flit'] for pid, info in pkt_infos.items()])
                report[layer_name] = int(total_flits)

        return report


    def calc_sram_power(self, benchmark_path):
        """Return a timeloop-style report of buffer actions. 
        """
        report = dict()

        layer_pattern = re.compile(r"^(.*)_\d+$")

        # get all layers
        all_layers = []
        with open(benchmark_path, 'r') as f:
            bm = yaml.load(f, Loader=yaml.FullLoader)
        for model_name, layers in bm.items():
            for layer in layers:
                for layer_name, core in layer.items():
                    all_layers.append(f"{layer_name}_{core}")


        layers_root = os.path.join(self.task_root, 'layers')
        for layer_dir in all_layers:
            layer_name = layer_pattern.match(layer_dir).group(1)
            layer_report = ['weight', 'input', 'output']
            layer_report = {k: {
                'instance': 0,
                'read': 0,
                'write': 0,
            } for k in layer_report}

            dump_mapping_path = os.path.join(layers_root, layer_dir, 'dump_mapping.yaml')
            with open(dump_mapping_path, 'r') as f:
                dump_mapping = yaml.load(f, Loader=yaml.FullLoader)

            converted_mapping = dict()
            for mapping_item in dump_mapping['mapping']:
                if not 'factors' in mapping_item.keys(): continue
                target = f"{mapping_item['target']}_{mapping_item['type']}"
                factors = mapping_item['factors'].split(" ")
                factors = [v.split("=") for v in factors]
                factors = {v[0]: int(v[1]) for v in factors}
                converted_mapping[target] = factors
            
            num_utilized_cores = np.prod(list(converted_mapping['DRAM_spatial'].values()))
            for k in layer_report.keys():
                layer_report[k]['instance'] = num_utilized_cores

            layer_report['weight']['read'] = np.prod([
                converted_mapping['DRAM_temporal']['C'],
                converted_mapping['DRAM_temporal']['M'],

                converted_mapping['GlobalBuffer_temporal']['C'],
                converted_mapping['GlobalBuffer_temporal']['M'],
                converted_mapping['GlobalBuffer_temporal']['R'],
                converted_mapping['GlobalBuffer_temporal']['S'],
                converted_mapping['GlobalBuffer_spatial']['C'],
                converted_mapping['GlobalBuffer_spatial']['M'],
                converted_mapping['PEAccuBuffer_spatial']['C'],
                converted_mapping['PEAccuBuffer_spatial']['M'],
            ])
            layer_report['weight']['write'] = layer_report['weight']['read']

            layer_report['input']['write'] = np.prod([
                converted_mapping['DRAM_temporal']['M'],
                converted_mapping['DRAM_temporal']['C'],
                converted_mapping['DRAM_temporal']['P'],
                converted_mapping['DRAM_temporal']['Q'],

                converted_mapping['GlobalBuffer_temporal']['P'] + converted_mapping['GlobalBuffer_temporal']['R'] - 1,
                converted_mapping['GlobalBuffer_temporal']['Q'] + converted_mapping['GlobalBuffer_temporal']['S'] - 1,
                converted_mapping['GlobalBuffer_temporal']['C'],
                converted_mapping['GlobalBuffer_spatial']['C'],
                converted_mapping['PEAccuBuffer_spatial']['C'],
            ])
            layer_report['input']['read'] = np.prod([
                layer_report['input']['write'],
                converted_mapping['GlobalBuffer_temporal']['M'],
            ])

            tmp_output_size = np.prod([
                converted_mapping['DRAM_temporal']['M'],
                converted_mapping['DRAM_temporal']['C'],
                converted_mapping['DRAM_temporal']['P'],
                converted_mapping['DRAM_temporal']['Q'],

                converted_mapping['GlobalBuffer_temporal']['M'],
                converted_mapping['GlobalBuffer_temporal']['P'],
                converted_mapping['GlobalBuffer_temporal']['Q'],
                converted_mapping['GlobalBuffer_spatial']['M'],
                converted_mapping['PEAccuBuffer_spatial']['M'],
            ])
            layer_report['output']['read'] = tmp_output_size * (converted_mapping['GlobalBuffer_temporal']['C'] - 1)
            layer_report['output']['write'] = tmp_output_size * converted_mapping['GlobalBuffer_temporal']['C']

            summary = {'read': 0, 'write': 0}
            for wr in summary.keys():
                for tensor_type in ['weight', 'input', 'output']:
                    summary[wr] += layer_report[tensor_type][wr] * layer_report[tensor_type]['instance']

            layer_report['summary'] = summary

            # JSON cannot serialize int64, dumbass, so I have to convert it myself
            for k in layer_report.keys():
                for l in layer_report[k].keys():
                    layer_report[k][l] = int(layer_report[k][l])
        
            report[layer_name] = layer_report
        return report

    def calc_reticle_power(self, full_benchmark_path):
        """Inter-reticle power
        Put layers in one reticle greedily until a reticle cannot fit.
        """

        def get_input_size(layer_name):
            model_pattern = re.compile(r"^(.*)_layer\d+$")
            model_name = model_pattern.match(layer_name).group(1)
            layer_spec_path = os.path.join(gc.database_root, model_name, f"{layer_name}.yaml")
            with open(layer_spec_path, 'r') as f:
                layer_spec = yaml.load(f, Loader=yaml.FullLoader)
            
            instance = layer_spec['problem']['instance']
            input_size = int(np.prod([instance[k] for k in ['C', 'P', 'Q']]))
            return input_size
            

        with open(full_benchmark_path, 'r') as f:
            bm = yaml.load(f, Loader=yaml.FullLoader)

        layer2core = dict()
        inter_reticle_transmission = dict()
        utilized_core_counter = 0
        model_names = []

        for model, layers in bm.items():
            model_names.append(model)
            for layer in layers:
                for layer_name, core in layer.items():
                    layer2core[layer_name] = core
        for layer_name, core in layer2core.items():
            if utilized_core_counter + core > self.core_array_h * self.core_array_w:
                utilized_core_counter = core
                inter_reticle_transmission[layer_name] = get_input_size(layer_name)
            else:
                utilized_core_counter += core
 
        return inter_reticle_transmission

class PowerAnalyzer():
    """Breakdown of power in each part
    """
    def __init__(self, **kwargs) -> None:
        self.frequency = 1e9
        self.bitwidth = 16

        self.mac_dynamic_energy = kwargs['mac_dynamic_energy']  # pJ (fp16)
        self.mac_static_power = kwargs['mac_static_power']      # W
        self.sram_static_power = kwargs['sram_static_power']    # W
        self.sram_read_energy = kwargs['sram_read_energy']      # pJ/bit
        self.sram_write_energy = kwargs['sram_write_energy']    # pJ/bit
        self.noc_static_power = kwargs['noc_static_power']      # W
        self.noc_channel_energy = kwargs['noc_channel_energy']  # pJ/bit
        self.reticle_channel_energy = kwargs['reticle_channel_energy']  # pJ/bit
        self.noc_bw = kwargs['noc_bw']
        self.reticle_bw = kwargs['reticle_bw']
        self.total_cores = kwargs['total_cores']
        self.core_num_mac = kwargs['core_num_mac']

    def run(self, prediction):
        report = dict()

        dynamic_scaling_factor = self.total_cores / self._get_mean_utilized_core(prediction)

        report['mac_dynamic'] = self.calc_mac_power(prediction) * dynamic_scaling_factor
        report['noc_dynamic'] = self.calc_noc_power(prediction) * dynamic_scaling_factor
        report['sram_dynamic'] = self.calc_sram_power(prediction) * dynamic_scaling_factor
        report['inter_reticle'] = self.calc_reticle_power(prediction) * dynamic_scaling_factor

        report['mac_static'] = self.mac_static_power * self.total_cores * self.core_num_mac
        report['noc_static'] = self.noc_static_power * self.total_cores
        report['sram_static'] = self.sram_static_power * self.total_cores
        report['total'] = np.sum([v for v in report.values()])

        return report

    def _get_mean_utilized_core(self, prediction):
        """Scale single batch power to full-wafer power.
        Run single batch, we utilize K cores on average, and consume power P
        Pipelineing multiple batches, we utilize all cores and consume power P'
        """
        latencies = []
        cores = []

        for layer in prediction['prediction'].keys():
            latencies.append(prediction['prediction'][layer] / self.frequency) 
            cores.append(prediction['power']['sram'][layer]['output']['instance'])

        total_latency = self._get_total_latency(prediction)
        weights = [v / total_latency for v in latencies]

        mean_utilized_core = np.average(cores, weights=weights)
        return mean_utilized_core


    def _get_total_latency(self, prediction):
        """Get total latency of current task.
        We only count selected layers, and normalized inter-reticle transmission's effect
        """
        selected_cycles = np.sum([v for k, v in prediction['prediction'].items()])

        inter_reticle_data = np.sum([v for k, v in prediction['power']['reticle'].items()])
        inter_reticle_cycles = inter_reticle_data / self.reticle_bw
        normalized_inter_reticle_cycles = inter_reticle_cycles * prediction['compute_percentage']

        total_latency = (selected_cycles + normalized_inter_reticle_cycles) / self.frequency
        return  total_latency


    def calc_mac_power(self, prediction):
        total_computation = np.sum([v for k, v in prediction['power']['mac'].items()])
        total_energy = total_computation * self.mac_dynamic_energy / 1e12
        total_latency = self._get_total_latency(prediction)
        mac_dynamic_power = total_energy / total_latency

        return mac_dynamic_power


    def calc_noc_power(self, prediction):
        total_flit = np.sum([v for k, v in prediction['power']['noc'].items()])
        total_energy = total_flit * self.noc_bw * self.noc_channel_energy / 1e12
        total_latency = self._get_total_latency(prediction)
        noc_dynamic_power = total_energy / total_latency
        return noc_dynamic_power

    def calc_sram_power(self, prediction):
        total_read_data = np.sum([v['summary']['read'] for k, v in prediction['power']['sram'].items()]) * self.bitwidth
        total_write_data = np.sum([v['summary']['write'] for k, v in prediction['power']['sram'].items()]) * self.bitwidth
        total_read_energy = total_read_data * self.sram_read_energy / 1e12
        total_write_energy = total_write_data * self.sram_write_energy / 1e12

        total_latency = self._get_total_latency(prediction)
        sram_dynamic_read = total_read_energy / total_latency
        sram_dynamic_write = total_write_energy / total_latency
        return sram_dynamic_read + sram_dynamic_write

    def calc_reticle_power(self, prediction):
        total_data = np.sum([v for k, v in prediction['power']['reticle'].items()]) * self.bitwidth
        total_energy = total_data * self.reticle_channel_energy  / 1e12
        total_latency = self._get_total_latency(prediction)
        reticle_dynamic_power = total_energy / total_latency
        return reticle_dynamic_power