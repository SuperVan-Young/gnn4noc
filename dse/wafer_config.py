import os
import time
import subprocess
import signal
import yaml
import json
from noc_spec import NoCSpec
import dse_global_control as gc

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.trace_parser.trace_parser import TraceParser
from dataset.predictor.lp_predictor import LinearProgrammingPredictor

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

    def run_focus(self, benchmark, run_timeloop=True, verbose=False, timeout=300):
        """Run focus for op_graph and routing
        """
        task_root = os.path.join(gc.dse_root, "tasks", f"{benchmark}_{self._get_config_briefing()}")
        if not os.path.exists(task_root):
            os.mkdir(task_root)

        benchmark_path = self._dump_benchmark(task_root, benchmark)

        if run_timeloop:
            self._dump_arch_config()

        focus_path = os.path.join(gc.focus_root, "focus.py")
        mode = "ted" if run_timeloop else "d"
        # array_size = max(self.core_array_h * self.reticle_array_h, self.core_array_w * self.reticle_array_w)
        array_size = max(self.core_array_h, self.core_array_w) * 2
        flit_size = self.core_noc_bw
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
        
        self.fetch_results(task_root)
        perf = self.predict_perf(task_root)
        return perf
            

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

    def fetch_results(self, task_root):
        assert os.path.exists(task_root)
        benchmark_path = os.path.join(task_root, f"benchmark.yaml")
        with open(benchmark_path, "r") as f:
            benchmark = yaml.load(f, Loader=yaml.FullLoader)
        array_size = max(self.core_array_h, self.core_array_w) * 2
        flit_size = self.core_noc_bw
        taskname = f"{list(benchmark.keys())[0]}_b1w{flit_size}_{array_size}x{array_size}"

        op_graph_path = gc.get_op_graph_path(taskname)
        routing_path = gc.get_routing_path(taskname)
        spec_path = gc.get_spec_path(taskname)

        assert op_graph_path != None
        assert routing_path != None
        assert spec_path != None

        os.system(f"cp {op_graph_path} {task_root}/op_graph.gpickle")
        os.system(f"cp {routing_path} {task_root}")
        os.system(f"cp {spec_path} {task_root}")

    def _dump_benchmark(self, task_root, benchmark_name):
        """Add suffix to the each model in benchmark config, return current benchmark path
        """
        benchmark_path = os.path.join(task_root, f"benchmark.yaml")
        benchmark_bu_path = os.path.join(gc.dse_root, "benchmark", f"{benchmark_name}.yaml")

        with open(benchmark_bu_path, 'r') as f:
            benchmark_bu = yaml.load(f, Loader=yaml.FullLoader)
        assert len(benchmark_bu) == 1, "WaferConfig: Not supporting multi model prediction"

        benchmark = {f"{benchmark_name}_{self._get_config_briefing()}" : benchmark_bu[benchmark_name]} 
        with open(benchmark_path, "w") as f:
            yaml.dump(benchmark, f)

        return benchmark_path

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
        pe_config[2]['attributes']['nbanks'] = sram_total_nbanks // 2
        pe_config[2]['attributes']['nports'] = 2  # read & write

        assert pe_config[0]['name'] == "PEInputBuffer"
        pe_config[0]['attributes']['word-bits'] = mac_datawidth
        pe_config[0]['attributes']['depth'] = sram_total_depth // 8
        pe_config[0]['attributes']['nbanks'] = sram_total_nbanks // 8

        assert pe_config[1]['name'] == "PEWeightBuffer"
        pe_config[1]['attributes']['word-bits'] = mac_datawidth
        pe_config[1]['attributes']['depth'] = sram_total_depth // 8
        pe_config[1]['attributes']['nbanks'] = sram_total_nbanks // 8

        assert pe_config[2]['name'] == "PEAccuBuffer"
        pe_config[2]['attributes']['word-bits'] = mac_datawidth
        pe_config[2]['attributes']['depth'] = 512 * 8 // mac_datawidth
        pe_config[2]['attributes']['nbanks'] = self.core_num_mac

        arch_path = os.path.join(gc.database_root, "arch/cerebras_like.yaml")
        with open(arch_path, 'w') as f:
            yaml.dump(arch_config, f)


    def predict_perf(self, task_root):
        graph_path = os.path.join(task_root, "op_graph.gpickle")
        routing_path = os.path.join(task_root, "routing_board")
        spec_path = os.path.join(task_root, "spatial_spec")
        
        trace_parser = TraceParser(
            graph_path=graph_path,
            outlog_path=None,
            routing_path=routing_path,
            spec_path=spec_path
        )

        core_array_size = max(self.core_array_h, self.core_array_w)
        noc_spec = NoCSpec(
            trace_parser=trace_parser,
            # core_array_h=self.core_array_h,
            core_array_h=core_array_size,
            # core_array_w=self.core_array_w,
            core_array_w=core_array_size,
            reticle_array_h=8,
            # reticle_array_h=self.reticle_array_h,
            reticle_array_w=8,
            # reticle_array_w=self.reticle_array_w,
            inter_reticle_bw=self.reticle_bw,
            inter_core_bw=self.core_noc_bw,
        )

        predictor = LinearProgrammingPredictor(trace_parser, noc_spec)
        total_latency = 0
        for layer_name in trace_parser.graph_parser.get_layers():
            total_latency += predictor.run(layer_name)

        prediction_path = os.path.join(task_root, "prediction.json")
        with open(prediction_path, "w") as f:
            json.dump(total_latency, f)

        return total_latency

if __name__ == "__main__":
    wafer_config = WaferConfig(
        core_num_mac = 16,
        core_buffer_bw = 256,
        core_buffer_size = 32,

        core_noc_bw = 1024,
        core_noc_vc = 4,
        core_noc_buffer_size = 2,
        core_array_h = 8,
        core_array_w = 8,

        reticle_bw = 1024,
        reticle_array_h = 4, 
        reticle_array_w = 4,

        wafer_mem_bw = 4096,
    )
    wafer_config.run_focus('gpt2-xl_tiny', True, True)