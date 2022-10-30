from math import log
import os
import time
import subprocess
import signal
import yaml
import json
import numpy as np

from noc_spec import NoCSpec
import dse_global_control as gc

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.trace_parser.trace_parser import TraceParser
from dataset.predictor.lp_predictor import LinearProgrammingPredictor

reticle_array_adjust = 1

def run_focus(benchmark_path, array_size, flit_size, mode, verbose=False, timeout=600):
    focus_path = os.path.join(gc.focus_root, "focus.py")

    command = f"python {focus_path} -bm {benchmark_path} -d {array_size} -b 1 \
                -fr {flit_size}-{flit_size}-{flit_size} {mode}" \
                # + " -debug" if verbose else ""

    begin_time = time.time()
    if verbose:
        sp = subprocess.Popen(command, shell=True, start_new_session=True)
    else:
        sp = subprocess.Popen(command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            shell=True, start_new_session=True)
    try:
        sp.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if verbose:
            print("Warning: running FOCUS timeout.")
        os.killpg(os.getpgid(sp.pid), signal.SIGTERM)
        sp.wait()

    end_time = time.time()
    if verbose:
        print(f"Info: running FOCUS complete in {end_time - begin_time} seconds.")

class WaferConfig():
    
    def __init__(self, **kwargs) -> None:
        self.core_num_mac = kwargs['core_num_mac']
        self.core_buffer_bw = kwargs['core_buffer_bw']
        self.core_buffer_size = kwargs['core_buffer_size']

        self.core_noc_bw = kwargs['core_noc_bw']
        self.core_noc_vc = kwargs['core_noc_vc']
        self.core_noc_buffer_size = kwargs['core_noc_buffer_size']
        self.core_array_h = kwargs['core_array_h']
        self.core_array_w = kwargs['core_array_w']

        self.reticle_bw = kwargs['reticle_bw']
        self.reticle_array_h = kwargs['reticle_array_h']
        self.reticle_array_w = kwargs['reticle_array_w']

        self.wafer_mem_bw = kwargs['wafer_mem_bw']

        assert self.core_num_mac >= 2, "core_num_mac < 2"

    def run_focus(self, benchmark, run_timeloop=True, verbose=False, timeout=600):
        """Run focus for op_graph and routing
        """
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
        # array_size = max(self.core_array_h * self.reticle_array_h, self.core_array_w * self.reticle_array_w)
        array_size = max(self.core_array_h, self.core_array_w) * reticle_array_adjust
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
                print(f"Error in predicting perf")
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

    def _dump_benchmark(self, task_root, benchmark_name):
        """Add suffix to the each model in benchmark config
        None of them will overlap with others results
        """
        benchmark_bu_path = os.path.join(gc.dse_root, "benchmark", f"{benchmark_name}.yaml")

        with open(benchmark_bu_path, 'r') as f:
            benchmark_bu = yaml.load(f, Loader=yaml.FullLoader)
        assert len(benchmark_bu) == 1, "WaferConfig: Not supporting multi model prediction"

        new_benchmark_names = []

        for factor in self._get_layer_scaling_factor(benchmark_name):

            benchmark_path = os.path.join(task_root, "benchmark", f"{benchmark_name}_cf{factor}.yaml")
            new_benchmark = [{list(l.keys())[0]: list(l.values())[0] * factor} for l in benchmark_bu[benchmark_name]]

            new_benchmark_name = f"{benchmark_name}_cf{factor}_{self._get_config_briefing()}"
            benchmark = {new_benchmark_name : new_benchmark} 
            with open(benchmark_path, "w") as f:
                yaml.dump(benchmark, f)

            new_benchmark_names.append(new_benchmark_name)
        
        return new_benchmark_names  # return for convenience

    def _get_layer_scaling_factor(self, benchmark_name):
        benchmark_bu_path = os.path.join(gc.dse_root, "benchmark", f"{benchmark_name}.yaml")

        with open(benchmark_bu_path, 'r') as f:
            benchmark_bu = yaml.load(f, Loader=yaml.FullLoader)
        assert len(benchmark_bu) == 1, "WaferConfig: Not supporting multi model prediction"

        reticle_core_num = self.core_array_h * self.core_array_w
        max_factor = reticle_core_num // np.sum([list(l.values())[0] for l in benchmark_bu[benchmark_name]])
        log_max_factor = int(log(max_factor, 2))
        factors = 2 ** np.arange(0, log_max_factor, log_max_factor / 4)
        
        for factor in factors:
            if factor > 4:
                factor = int(factor // 4 * 4)
            else:
                factor = int(factor)

        return set(factors)

    def _dump_arch_config(self):
        arch_path = os.path.join(gc.database_root, "arch_bu/cerebras_like.yaml")
        with open(arch_path, 'r') as f:
            arch_config = yaml.load(f, Loader=yaml.FullLoader)

        mac_datawidth = 16
        sram_total_depth = self.core_buffer_size * 1024 // mac_datawidth  # assume only one word in a row
        sram_total_nbanks = self.core_buffer_bw // mac_datawidth  # use nbanks to control bw

        # mac number
        pe_config = arch_config['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local']
        assert pe_config[-1]['name'] == "LMAC[0..3]"
        pe_config[-1]['name'] = f"LMAC[0..{self.core_num_mac-1}]"
        pe_config[-1]['attributes']['datawidth'] = mac_datawidth
        assert pe_config[-2]['name'] == "PEWeightRegs[0..3]"
        pe_config[-2]['name'] = f"PEWeightRegs[0..{self.core_num_mac-1}]"
        pe_config[-2]['attributes']['word-bits'] = mac_datawidth


        # buffer allocation:
        # - global_buffer: 1/2 size & bandwidth
        # - weight & input buffer : 1/4 size & bandwidth, 1/8 each
        # - accum buffer: 512B, 16 * mac bit/cycle
        arch_config['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['name'] = 'PE[0..1]' # stay fanout ...
        ws_config = arch_config['architecture']['subtree'][0]['subtree'][0]['local']

        assert ws_config[0]['name'] == "GlobalBuffer"
        ws_config[0]['attributes']['word-bits'] = mac_datawidth
        ws_config[0]['attributes']['depth'] = sram_total_depth // 2
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
        core_num_mac = 16,
        core_buffer_bw = 256,
        core_buffer_size = 32,

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
    # wafer_config.predict_perf("/home/xuechenhao/gnn4noc/dse/tasks1030/gpt2-xl_tiny_cm16_cbw256_csz32_nvc4_nbw1024_nsz2_nah50_naw50_rbw1024_rah4_raw4_wmbw4096", "gpt2-xl_tiny_cf1_cm16_cbw256_csz32_nvc4_nbw1024_nsz2_nah50_naw50_rbw1024_rah4_raw4_wmbw4096")