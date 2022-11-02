import os
import yaml
import json
import re
import numpy as np
from tqdm import tqdm
import time
import multiprocessing as mp

from noc_spec import NoCSpec
import dse_global_control as gc
from sp_runner import run_focus, run_timeloop_mapper

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.trace_parser.trace_parser import TraceParser
from dataset.predictor.lp_predictor import LinearProgrammingPredictor


class WaferConfig():
    
    def __init__(self, **kwargs) -> None:
        # core config
        self.core_num_mac = kwargs['core_num_mac']
        self.core_buffer_bw = kwargs['core_buffer_bw']
        self.core_buffer_size = kwargs['core_buffer_size']

        # noc config
        self.core_noc_bw = kwargs['core_noc_bw']
        self.core_noc_vc = kwargs['core_noc_vc']
        self.core_noc_buffer_size = kwargs['core_noc_buffer_size']
        self.core_array_h = kwargs['core_array_h']
        self.core_array_w = kwargs['core_array_w']

        # reticle config
        self.reticle_bw = kwargs['reticle_bw']
        self.reticle_array_h = kwargs['reticle_array_h']
        self.reticle_array_w = kwargs['reticle_array_w']

        # memory config
        self.wafer_mem_bw = kwargs['wafer_mem_bw']

        assert self.core_num_mac >= 2, "core_num_mac < 2"

        self.task_root = os.path.join(gc.task_root, self._get_config_briefing())

    def run(self, run_timeloop=True):
        """Run focus toolchain
        """
        #TODO: benchmark: default is all benchmarks, but could specify one task
        task_root = os.path.join(gc.task_root, self._get_config_briefing())
        if not os.path.exists(task_root):
            os.mkdir(task_root)

        # dump all benchmarks to task_root/benchmark/xxx.yaml
        self._dump_benchmark()

        if run_timeloop:
            self._dump_arch_config()
            self._dump_constraints_config()
            self._dump_modified_arch_config()

            for layers_root, dirs, files in os.walk(os.path.join(task_root, "layers")):
                layers = [os.path.join(layers_root, l) for l in dirs]
                with mp.Pool(processes=4) as pool:  # this is much faster
                    pool.map(run_timeloop_mapper, layers)
                break

        self.predict_perf()

    def _get_config_briefing(self):
        briefs = [
            ('cm', self.core_num_mac),
            ('cbw', self.core_buffer_bw),
            ('csz', self.core_buffer_size),
            ('nvc', self.core_noc_vc),
            ('nbw', self.core_noc_bw),
            ('nsz', self.core_noc_buffer_size),
            ('nah', self.core_array_h),
            ('naw', self.core_array_w),
            ('rbw', self.reticle_bw),
            ('rah', self.reticle_array_h),
            ('raw', self.reticle_array_w),
            ('wmbw', self.wafer_mem_bw),
        ]
        return "_".join([f"{k}{v}" for k, v in briefs])

    def _dump_benchmark(self):
        """Dump benchmark config
        - Rename benchmark model, s.t. FOCUS will not overlap the result
        - Adjust core allocation, try to give as many cores as possible
        """
        benchmark_root = os.path.join(self.task_root, "benchmark")
        if not os.path.exists(benchmark_root):
            os.mkdir(benchmark_root)

        for benchmark_bu_root, dirs, files in os.walk(os.path.join(gc.dse_root, "benchmark")):
            for file in files:
                benchmark_bu_path = os.path.join(benchmark_bu_root, file)

                with open(benchmark_bu_path, 'r') as f:
                    benchmark_bu = yaml.load(f, Loader=yaml.FullLoader)
                assert len(benchmark_bu) == 1, "WaferConfig: only support single model performance prediction"
                for k, v in benchmark_bu.items():
                    benchmark_bu_name, benchmark_bu_layers = k, v

                get_layer_name = lambda x: list(x.keys())[0]
                get_layer_num_core = lambda x: list(x.values())[0]

                # core factor
                available_core_num = self.core_array_h * self.core_array_w
                demanding_core_num = np.sum([get_layer_num_core(l) for l in benchmark_bu_layers])
                core_factor = available_core_num / demanding_core_num

                new_benchmark_name = f"{benchmark_bu_name}_{self._get_config_briefing()}"
                new_benchmark_config = [{get_layer_name(l): max(int(get_layer_num_core(l) * core_factor), 2)} for l in benchmark_bu_layers]
                new_benchmark = {new_benchmark_name : new_benchmark_config}

                benchmark_path = os.path.join(benchmark_root, file)
                with open(benchmark_path, "w") as f:
                    yaml.dump(new_benchmark, f)

    def _dump_arch_config(self):
        """Modify architecture configuration.
        """
        arch_path = os.path.join(gc.database_root, "arch_bu/cerebras_like.yaml")
        with open(arch_path, 'r') as f:
            arch_config = yaml.load(f, Loader=yaml.FullLoader)

        mac_datawidth = 16
        sram_total_depth = self.core_buffer_size * 1024 // mac_datawidth  # assume only one word in a row
        sram_total_nbanks = self.core_buffer_bw // mac_datawidth  # use nbanks to control bw

        # mac number
        # keep 2 PE for fanout, each PE should have 1/2 mac & reg
        pe_config = arch_config['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local']
        assert pe_config[-1]['name'] == "LMAC[0..3]"
        pe_config[-1]['name'] = f"LMAC[0..{(self.core_num_mac-1) // 2}]"
        pe_config[-1]['attributes']['datawidth'] = mac_datawidth
        assert pe_config[-2]['name'] == "PEWeightRegs[0..3]"
        pe_config[-2]['name'] = f"PEWeightRegs[0..{(self.core_num_mac-1) // 2}]"
        pe_config[-2]['attributes']['word-bits'] = mac_datawidth


        # buffer allocation:
        # keep GB for timeloop, assume size(GB) = size(WB) + size(IB)
        # - global_buffer: 1 size & bandwidth
        # - weight & input buffer : 1/2 size & bandwidth, 1/4 each
        # - accum buffer: 512B, 16 * mac bit/cycle, for free
        arch_config['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['name'] = 'PE[0..1]' # stay fanout ...
        ws_config = arch_config['architecture']['subtree'][0]['subtree'][0]['local']

        assert ws_config[0]['name'] == "GlobalBuffer"
        ws_config[0]['attributes']['word-bits'] = mac_datawidth
        ws_config[0]['attributes']['depth'] = sram_total_depth
        pe_config[2]['attributes']['nbanks'] = sram_total_nbanks
        pe_config[2]['attributes']['nports'] = 2  # read & write

        assert pe_config[0]['name'] == "PEInputBuffer"
        pe_config[0]['attributes']['word-bits'] = mac_datawidth
        pe_config[0]['attributes']['depth'] = sram_total_depth // 4
        pe_config[0]['attributes']['nbanks'] = sram_total_nbanks // 4

        assert pe_config[1]['name'] == "PEWeightBuffer"
        pe_config[1]['attributes']['word-bits'] = mac_datawidth
        pe_config[1]['attributes']['depth'] = sram_total_depth // 4
        pe_config[1]['attributes']['nbanks'] = sram_total_nbanks // 4

        assert pe_config[2]['name'] == "PEAccuBuffer"
        pe_config[2]['attributes']['word-bits'] = mac_datawidth
        pe_config[2]['attributes']['depth'] = 512 * 8 // mac_datawidth
        pe_config[2]['attributes']['nbanks'] = self.core_num_mac

        # dump arch config
        arch_root = os.path.join(self.task_root, "arch")
        if not os.path.exists(arch_root):
            os.mkdir(arch_root)

        arch_path = os.path.join(arch_root, "cerebras_like.yaml")
        with open(arch_path, 'w') as f:
            yaml.dump(arch_config, f)

    def _dump_constraints_config(self, verbose=False):
        """Add constraints for faster timeloop searching"""

        def get_max_factor(num, bound):
            """Return maximum i, s.t. num % i == 0, i <= bound
            """
            if bound > np.sqrt(num):
                for i in range(int(np.ceil(num / bound)), int(np.ceil(np.sqrt(num)))):
                    if num % i == 0:
                        return num // i
                for i in range(int(np.sqrt(num)), 0, -1):
                    if num % i == 0:
                        return i
            else:
                for i in range(bound, 0, -1):
                    if num % i == 0:
                        return i

        def get_max_subfactor(num, factor):
            """find maximum i, s.t. factor % i == 0, num % i == 0
            """
            for i in range(1, int(np.ceil(np.sqrt(factor)))):
                if factor % i == 0:
                    i_ = factor // i
                    if num % i_ == 0:
                        return i_
            for i in range(int(np.sqrt(factor)), 0, -1):
                if factor % i == 0 and num % i == 0:
                    return i

        def get_unroll_factors(dims, factor):
            """Unroll dims, w.r.t. priority
            """
            factors = []
            for i, dim in enumerate(dims):
                unroll_factor = get_max_subfactor(dim, factor)
                # if verbose: print(f"{unroll_factor} = get_max_subfactor({dim}, {factor})")
                factors.append(unroll_factor)
                factor = factor // unroll_factor
            return factors
        
        layers_root = os.path.join(self.task_root, "layers")
        if not os.path.exists(layers_root):
            os.mkdir(layers_root)

        for benchmark_root, dirs, files in os.walk(os.path.join(self.task_root, "benchmark")):
            for file in files:
                benchmark_path = os.path.join(benchmark_root, file)
                with open(benchmark_path, 'r') as f:
                    benchmark = yaml.load(f, Loader=yaml.FullLoader)
                for k, v in benchmark.items():
                    benchmark_name, benchmark_layers = k, v

                get_layer_name = lambda x: list(x.keys())[0]
                get_layer_num_core = lambda x: list(x.values())[0]

                for l in benchmark_layers:
                    num_core = get_layer_num_core(l)

                    parsed_layer_name = re.search(r"(^.*)_layer(\d+)$", get_layer_name(l))
                    model_name, layer_id = parsed_layer_name.group(1), parsed_layer_name.group(2)
                    layer_config_path = os.path.join(gc.focus_root, "database", model_name, f"{model_name}_layer{layer_id}.yaml")
                    with open(layer_config_path, 'r') as f:
                        layer_config = yaml.load(f, Loader=yaml.FullLoader)
                    C = layer_config['problem']['instance']['C']
                    M = layer_config['problem']['instance']['M']
                    P = layer_config['problem']['instance']['P']
                    Q = layer_config['problem']['instance']['Q']
                    R = layer_config['problem']['instance']['R']
                    S = layer_config['problem']['instance']['S']

                    if verbose:
                        print(f"layer name: {get_layer_name(l)}")
                        print(f"Initial C, M, P, Q, R, S = {C}, {M}, {P}, {Q}, {R}, {S}")

                    constraint_bu_path = os.path.join(gc.focus_root, "database", "constraints", "simba_constraints_copy.yaml")
                    with open(constraint_bu_path, 'r') as f:
                        constraint = yaml.load(f, Loader=yaml.FullLoader)
                        constraint_targets = constraint['mapspace_constraints']['targets']

                    # utilize PE first, only unroll C, M
                    num_utilized_mac = get_max_factor(C*M, self.core_num_mac)
                    if verbose:
                        print(f"utilized mac: {num_utilized_mac} / {self.core_num_mac}")
                    if num_utilized_mac % 2 == 0:
                        num_utilized_mac = num_utilized_mac // 2
                        if C % 2 == 0:
                            C = C // 2
                            constraint_targets.append({
                                'target': 'GlobalBuffer',
                                'type': 'spatial',
                                'factors': 'C=2',
                            })
                        else:
                            M = M // 2
                            constraint_targets.append({
                                'target': 'GlobalBuffer',
                                'type': 'spatial',
                                'factors': 'M=2',
                            })

                    # for mac, first unroll C, then unroll M
                    mac_dims = [C, M]
                    mac_factors = get_unroll_factors(mac_dims, num_utilized_mac)
                    assert(np.prod(mac_factors) == num_utilized_mac)
                    constraint_targets.append({
                        'target': 'PEAccuBuffer',
                        'type': 'spatial',
                        'factors': " ".join([f"{l}={factor}" for l, factor in zip(['C', 'M'], mac_factors)]),
                    })
                    C = C // mac_factors[0]
                    M = M // mac_factors[1]

                    # utilize all cores, unroll C, M, P, Q, R, S
                    num_utilized_core = get_max_factor(C*M*P*Q*R*S, num_core)
                    if verbose:
                        print(f"utilized core: {num_utilized_core} / {num_core}")
                    core_dims = [C, M, P, Q, R, S]
                    core_factors = get_unroll_factors(core_dims, num_utilized_core)
                    assert(np.prod(core_factors) == num_utilized_core)
                    constraint_targets.append({
                        'target': 'DRAM',
                        'type': 'spatial',
                        'factors': " ".join([f"{l}={factor}" for l, factor in zip(['C', 'M', 'P', 'Q', 'R', 'S'], core_factors)]),
                    })

                    layer_root = os.path.join(layers_root,  f"{get_layer_name(l)}_{get_layer_num_core(l)}")
                    if not os.path.exists(layer_root):
                        os.mkdir(layer_root)

                    constraint_path = os.path.join(layer_root, "constraints.yaml")
                    with open(constraint_path, 'w') as f:
                        yaml.dump(constraint, f)
                    
                    if verbose:
                        print("core factors: ", " ".join([f"{l}={factor}" for l, factor in zip(['C', 'M', 'P', 'Q', 'R', 'S'], core_factors)]))
                        print("mac factors: ", " ".join([f"{l}={factor}" for l, factor in zip(['C', 'M'], mac_factors)]))
                        print()

    def _dump_modified_arch_config(self):
        """Copied from FOCUS toolchain.
        """
        with open(os.path.join(self.task_root, 'arch', 'cerebras_like.yaml')) as f:
            arch = yaml.load(f, Loader=yaml.FullLoader)

        for layers_root, dirs, files in os.walk(os.path.join(self.task_root, "layers")):
            for layer_dir in dirs:
                top_level_pe_cnt = int(re.search(r"^.*_(\d+)$", layer_dir).group(1))
                top_level_name = arch["architecture"]["subtree"][0]["subtree"][0]["name"]
                new_top_level_name = re.sub(r"0..\d+", "0.."+str(top_level_pe_cnt-1), top_level_name)
                arch["architecture"]["subtree"][0]["subtree"][0]["name"] = new_top_level_name

                new_top_arch_spec = os.path.join(layers_root, layer_dir, "modified_arch.yaml")
                with open(new_top_arch_spec, 'w') as f:
                    yaml.dump(arch, f)


    def predict_perf(self):
        os.system(f"cp {os.path.join(self.task_root, 'arch', 'cerebras_like.yaml')} {os.path.join(gc.database_root, 'arch')}")
        prediction_root = os.path.join(self.task_root, "prediction")
        if not os.path.exists(prediction_root):
            os.mkdir(prediction_root)

        for benchmark_root, __, files in os.walk(os.path.join(self.task_root, 'benchmark')):
            for file in files:
                benchmark_path = os.path.join(benchmark_root, file)

                with open(benchmark_path, 'r') as f:
                    benchmark = yaml.load(f, Loader=yaml.FullLoader)
                assert len(benchmark) == 1, "WaferConfig: only support single model performance prediction"
                for k, v in benchmark.items():
                    benchmark_name, benchmark_layers = k, v

                get_layer_name = lambda x: list(x.keys())[0]
                get_layer_num_core = lambda x: list(x.values())[0]

                # copy timeloop mapper results to FOCUS dir``
                layer_dirs = [f"{get_layer_name(l)}_{get_layer_num_core(l)}" for l in benchmark_layers]
                for layer_dir in layer_dirs:
                    os.system(f"cp -r {os.path.join(self.task_root, 'layers', layer_dir)} {os.path.join(gc.focus_root, 'buffer', 'timeloop-512g', layer_dir)}") 

                mode = "ed"  # communication still use FOCUS'
                core_array_size = max(self.core_array_h, self.core_array_w)
                flit_size = self.core_noc_bw
                run_focus(benchmark_path, core_array_size, flit_size, mode, verbose=True, debug=True, timeout=300)

                taskname = f"{benchmark_name}_b1w{flit_size}_{core_array_size}x{core_array_size}"
                graph_path = gc.get_op_graph_path(taskname)
                routing_path = gc.get_routing_path(taskname)
                spec_path = gc.get_spec_path(taskname)
                assert graph_path != None
                assert routing_path != None
                assert spec_path != None
                trace_parser = TraceParser(
                    graph_path=graph_path,
                    outlog_path=None,
                    routing_path=routing_path,
                    spec_path=spec_path
                )

                noc_spec = NoCSpec(
                    trace_parser=trace_parser,
                    core_array_h=core_array_size,
                    core_array_w=core_array_size,
                    reticle_array_h=1,
                    reticle_array_w=1,
                    inter_reticle_bw=self.reticle_bw,
                    inter_core_bw=self.core_noc_bw,
                )

                predictor = LinearProgrammingPredictor(trace_parser, noc_spec)
                latencies = dict()
                for layer_name in trace_parser.graph_parser.get_layers():
                    latencies[layer_name] = int(predictor.run(layer_name))

                prediction_path = os.path.join(prediction_root, f"{benchmark_name}.json")
                with open(prediction_path, "w") as f:
                    json.dump(latencies, f)
            break

if __name__ == "__main__":
    wafer_config = WaferConfig(
        core_num_mac = 256,
        core_buffer_bw = 256,
        core_buffer_size = 1024,

        core_noc_bw = 1024,
        core_noc_vc = 4,
        core_noc_buffer_size = 2,
        core_array_h = 50,
        core_array_w = 50,

        reticle_bw = 1024,
        reticle_array_h = 4, 
        reticle_array_w = 4,

        wafer_mem_bw = 4096, # testing!
    )
    wafer_config.run(run_timeloop=False)