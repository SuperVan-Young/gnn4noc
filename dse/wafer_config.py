import os
import yaml
import json
import numpy as np

from noc_spec import NoCSpec
import dse_global_control as gc
from sp_runner import run_focus

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

    def run(self, benchmark_name, run_timeloop=True, verbose=False, timeout=600):
        """Run focus toolchain
        """
        #TODO: benchmark: default is all benchmarks, but could specify one task
        task_root = os.path.join(gc.task_root, f"{benchmark}_{self._get_config_briefing()}")
        if not os.path.exists(task_root):
            os.mkdir(task_root)
            os.mkdir(os.path.join(task_root, "benchmark"))
            os.mkdir(os.path.join(task_root, "prediction"))

        # dump all benchmarks to task_root/benchmark/xxx.yaml
        benchmark_names = self._dump_benchmark(task_root, benchmark)

        if run_timeloop:
            self._dump_arch_config()

        mode = "ted" if run_timeloop else "d"
        array_size = max(self.core_array_h, self.core_array_w)
        flit_size = self.core_noc_bw

        for root, __, files in os.walk(os.path.join(task_root, 'benchmark')):
            for file in files:
                benchmark_path = os.path.join(root, file)
                run_focus(benchmark_path, array_size, flit_size, mode, verbose, timeout)
            break
        
        for benchmark_name in benchmark_names:
            try:
                self.predict_perf(task_root, benchmark_name)
            except:
                print(f"Error in predicting perf: {benchmark_name} {self._get_config_briefing()}")
                continue

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
        for benchmark_bu_root, dirs, files in os.walk(os.path.join(gc.dse_root, "benchmark")):
            for file in files:
                benchmark_bu_path = os.path.join(benchmark_bu_root, file)

                with open(benchmark_bu_path, 'r') as f:
                    benchmark_bu = yaml.load(f, Loader=yaml.FullLoader)
                assert len(benchmark_bu) == 1, "WaferConfig: only support single model performance prediction"
                for k, v in benchmark_bu:
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

                benchmark_path = os.path.join(self.task_root, "benchmark", file)
                with open(benchmark_path, "w") as f:
                    yaml.dump(new_benchmark, f)
        

    def _get_layer_allocation(self, benchmark_name):
        """Allocate all cores on a reticle to each layer in benchmark model
        """
        benchmark_bu_path = os.path.join(gc.dse_root, "benchmark", f"{benchmark_name}.yaml")

        with open(benchmark_bu_path, 'r') as f:
            benchmark_bu = yaml.load(f, Loader=yaml.FullLoader)
        assert len(benchmark_bu) == 1, "WaferConfig: Not supporting multi model prediction"

        reticle_core_num = self.core_array_h * self.core_array_w
        max_factor = reticle_core_num / 
        if max_factor < 1:
            print("Wafer Config: warning: max factor < 1, the array is over-utilized. Reset max factor to 1 by default.")
        max_factor = max(max_factor, 1)  # at least 1

        return max_factor

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

        arch_path = os.path.join(gc.database_root, "arch/cerebras_like.yaml")
        with open(arch_path, 'w') as f:
            yaml.dump(arch_config, f)

    def _dump_constraints_config(self):
        """Add constraints for faster timeloop searching"""


    def predict_perf(self, task_root, benchmark_name):
        array_size = max(self.core_array_h, self.core_array_w) * reticle_array_adjust
        flit_size = self.core_noc_bw
        taskname = f"{benchmark_name}_b1w{flit_size}_{array_size}x{array_size}"

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

        # assume that we could put a model inside a reticle
        core_array_size = max(self.core_array_h, self.core_array_w)
        noc_spec = NoCSpec(
            trace_parser=trace_parser,
            # core_array_h=self.core_array_h,
            core_array_h=core_array_size,
            # core_array_w=self.core_array_w,
            core_array_w=core_array_size,
            reticle_array_h=reticle_array_adjust,
            # reticle_array_h=self.reticle_array_h,
            reticle_array_w=reticle_array_adjust,
            # reticle_array_w=self.reticle_array_w,
            inter_reticle_bw=self.reticle_bw,
            inter_core_bw=self.core_noc_bw,
        )

        predictor = LinearProgrammingPredictor(trace_parser, noc_spec)
        latencies = dict()
        for layer_name in trace_parser.graph_parser.get_layers():
            latencies[layer_name] = int(predictor.run(layer_name))

        prediction_path = os.path.join(task_root, "prediction", f"{benchmark_name}.json")
        with open(prediction_path, "w") as f:
            json.dump(latencies, f)

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

        wafer_mem_bw = 4096,
    )
    wafer_config.run_focus('gpt2-xl_tiny', True, True)