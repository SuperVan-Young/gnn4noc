import os
import yaml
import json
import re
import numpy as np
from tqdm import tqdm
import time
from copy import deepcopy
import multiprocessing as mp
from itertools import product

from noc_spec import NoCSpec
import dse_global_control as gc
from sp_runner import run_focus, run_timeloop_mapper, run_timeloop_model
from scheduling import UnrollingConstraint
from power import PowerPredictor

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

    def run(self, dump_benchmark, invoke_timeloop_mapper, invoke_timeloop_model, invoke_focus, predict, verbose=False):
        """Run focus toolchain
        """
        task_root = os.path.join(gc.task_root, self._get_config_briefing())
        if not os.path.exists(task_root):
            os.mkdir(task_root)

        # dump all necessary configuration specs
        if dump_benchmark:
            self._dump_benchmark()
            self._dump_arch_config()
            self._dump_constraints_config()
            self._dump_modified_arch_config()
            print(f"{self._get_config_briefing()}: Finished dumping benchmark")

        if invoke_timeloop_mapper:
            if verbose: print(f"{self._get_config_briefing()}: Running timeloop mapper")

            for layers_root, dirs, files in os.walk(os.path.join(task_root, "layers")):
                layers = [os.path.join(layers_root, l) for l in dirs]
                with mp.Pool(processes=8) as pool:  # this is much faster
                    pool.map(run_timeloop_mapper, layers)
                break
            if verbose: print(f"{self._get_config_briefing()}: Finish timeloop mapper")
        
        if invoke_timeloop_model:
            if verbose: print(f"{self._get_config_briefing()}: Running timeloop model")
            for layers_root, dirs, files in os.walk(os.path.join(task_root, "layers")):
                layers = [os.path.join(layers_root, l) for l in dirs]
                with mp.Pool(processes=32) as pool:
                    pool.map(run_timeloop_model, layers)
                break
            if verbose: print(f"{self._get_config_briefing()}: Finish timeloop model")

        if predict:
            if verbose: print(f"{self._get_config_briefing()}: Predicting performance")
            self.predict_perf(invoke_focus)
            if verbose: print(f"{self._get_config_briefing()}: Finish predicting performance")

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

        benchmark_full_root = os.path.join(self.task_root, "benchmark_full")  # save for inter-reticle power estimation
        if not os.path.exists(benchmark_full_root):
            os.mkdir(benchmark_full_root)

        for benchmark_bu_root, dirs, files in os.walk(os.path.join(gc.dse_root, "benchmark")):
            for file in files:
                benchmark_bu_path = os.path.join(benchmark_bu_root, file)
                benchmark_constraint_path = os.path.join(os.path.join(gc.dse_root, "benchmark_constraint", file))
                assert os.path.exists(benchmark_constraint_path), f"Benchmark constraint {benchmark_constraint_path} should exist!"

                with open(benchmark_bu_path, 'r') as f:
                    benchmark_bu = yaml.load(f, Loader=yaml.FullLoader)
                assert len(benchmark_bu) == 1, "WaferConfig: only support single model performance prediction"
                for k, v in benchmark_bu.items():
                    benchmark_bu_name, benchmark_bu_layers = k, v

                get_layer_name = lambda x: list(x.keys())[0]
                get_layer_num_core = lambda x: list(x.values())[0]

                # core factor
                available_core_num = self.core_array_h * self.core_array_w * self.reticle_array_h * self.reticle_array_w
                demanding_core_num = np.sum([get_layer_num_core(l) for l in benchmark_bu_layers])
                core_factor = available_core_num / demanding_core_num
                core_factor = core_factor / gc.num_effective_model

                new_benchmark_name = f"{benchmark_bu_name}_{self._get_config_briefing()}"
                new_benchmark_config = [{get_layer_name(l): min(max(int(get_layer_num_core(l) * core_factor), 2), self.core_array_h * self.core_array_w)} for l in benchmark_bu_layers]

                # save for inter-reticle power estimation
                benchmark_full_path = os.path.join(benchmark_full_root, file)
                new_benchmark_full = {new_benchmark_name : new_benchmark_config}
                with open(benchmark_full_path, 'w') as f:
                    yaml.dump(new_benchmark_full, f)

                # delete layers not in constraint
                with open(benchmark_constraint_path, "r") as f:
                    benchmark_constraint = yaml.load(f, Loader=yaml.FullLoader)
                assert len(benchmark_constraint) == 1, "WaferConfig: benchmark constraint multi model!"
                for k, v in benchmark_constraint.items():
                    benchmark_constraint = v
                    break
                new_benchmark_config = [l for l in new_benchmark_config if get_layer_name(l) in benchmark_constraint]

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
        """Add constraints for faster timeloop searching
        Cannot assure a valid mapping. FXXK TIMELOOP"""
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

                    unroller = UnrollingConstraint(
                        num_core=num_core,
                        core_num_mac=self.core_num_mac,
                        core_buffer_size=self.core_buffer_size,
                        core_noc_bw=self.core_noc_bw,
                        N=layer_config['problem']['instance']['N'],
                        C=layer_config['problem']['instance']['C'],
                        M=layer_config['problem']['instance']['M'],
                        P=layer_config['problem']['instance']['P'],
                        Q=layer_config['problem']['instance']['Q'],
                        R=layer_config['problem']['instance']['R'],
                        S=layer_config['problem']['instance']['S'],
                    )

                    constraint_bu_path = os.path.join(gc.focus_root, "database", "constraints", "simba_constraints_copy.yaml")
                    with open(constraint_bu_path, 'r') as f:
                        constraint = yaml.load(f, Loader=yaml.FullLoader)
                    constraint['mapspace_constraints']['targets'] = unroller.run()

                    layer_root = os.path.join(layers_root,  f"{get_layer_name(l)}_{get_layer_num_core(l)}")
                    if not os.path.exists(layer_root):
                        os.mkdir(layer_root)

                    constraint_path = os.path.join(layer_root, "constraints.yaml")
                    with open(constraint_path, 'w') as f:
                        yaml.dump(constraint, f)

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


    def predict_perf(self, invoke_focus):
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

                mode = "d"  # communication still use FOCUS'
                # core_array_size = max(self.core_array_h, self.core_array_w) * max(self.reticle_array_h, self.reticle_array_w)
                core_array_size = int(np.sqrt(np.sum([get_layer_num_core(l) for l in benchmark_layers]))) + 2  # eee, cannot run too big
                flit_size = self.core_noc_bw
                timeloop_buffer_path = os.path.join(self.task_root, "layers")
                if invoke_focus: run_focus(benchmark_path, core_array_size, flit_size, mode, timeloop_buffer_path, verbose=False, debug=False, timeout=3600)

                # trace parser
                taskname = f"{benchmark_name}_b1w{flit_size}_{core_array_size}x{core_array_size}"
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

                perf_predictor = LinearProgrammingPredictor(trace_parser, None)
                report = {
                    "prediction": {},
                    "computation": {},
                    "transmission": {},
                }
                for layer_name in trace_parser.graph_parser.get_layers():
                    report["prediction"][layer_name] = int(perf_predictor.run(layer_name))
                    report["computation"][layer_name] = int(perf_predictor.get_computation(layer_name))
                    report['transmission'][layer_name] = perf_predictor.get_data_transmission(layer_name)

                power_predictor = PowerPredictor(
                    task_root = self.task_root,
                    core_array_size=core_array_size,
                    flit_size = flit_size,
                    core_array_h = self.core_array_h,
                    core_array_w = self.core_array_w,
                )
                power_report = power_predictor.run(
                    benchmark_path=benchmark_path,
                    full_benchmark_path=os.path.join(self.task_root, 'benchmark_full', file),
                )
                report['power'] = power_report

                prediction_path = os.path.join(prediction_root, f"{benchmark_name}.json")
                with open(prediction_path, "w") as f:
                    f.write(json.dumps(report, indent=4))
            break

if __name__ == "__main__":
    wafer_config = WaferConfig(
        core_num_mac = 64,
        core_buffer_bw = 2048,
        core_buffer_size = 512,

        core_noc_bw = 4096,
        core_noc_vc = 4,
        core_noc_buffer_size = 4,
        core_array_h = 18,
        core_array_w = 19,

        reticle_bw = 1,
        reticle_array_h = 7, 
        reticle_array_w = 8,

        wafer_mem_bw = 4096, # testing!
    )
    wafer_config.run(dump_benchmark=True, invoke_timeloop_mapper=True, invoke_timeloop_model=True, invoke_focus=True, predict=True, verbose=True)