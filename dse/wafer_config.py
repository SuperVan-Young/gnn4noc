import os
import time
import subprocess
import signal
import yaml
import json
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
        assert self.core_array_h == self.core_array_w, "core array h != w"
        assert self.reticle_array_h == self.core_array_w, "reticle array h != w"

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
        array_size = self.core_array_h * self.reticle_array_h
        flit_size = self.core_noc_bw
        command = f"python {focus_path} -bm {benchmark_path} -d {array_size} -b 1 \
                    -fr {flit_size}-{flit_size}-{flit_size} {mode}"
    
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
        self.predict_perf(task_root)
            

    def _get_config_briefing(self):
        briefs = {
            'cm': self.core_num_mac,
            'cbw': self.core_buffer_bw,
            'csz': self.core_buffer_size,
            'nvc': self.core_noc_vc,
            'nbw': self.core_noc_bw,
            'nsz': self.core_noc_buffer_size,
            'nah': self.core_array_h,
            'naw': self.core_array_w,
            'rbw': self.reticle_bw,
            'rah': self.reticle_array_h,
            'raw': self.reticle_array_w,
            'wmbw' : self.wafer_mem_bw,
        }
        return "_".join({f"{k}-{v}" for k, v in briefs.items()})

    def fetch_result(self, task_root):
        assert os.path.exists(task_root)
        benchmark_path = os.path.join(task_root, f"benchmark.yaml")
        with open(benchmark_path, "r") as f:
            benchmark = yaml.load(f, Loader=yaml.FullLoader)
        array_size = self.core_array_h * self.reticle_array_h
        flit_size = self.core_noc_bw
        taskname = f"{benchmark.keys()[0]}_b1w{flit_size}_{array_size}x{array_size}"

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
            benchmark_bu = yaml.load(benchmark_bu_path, Loader=yaml.FullLoader)
        assert len(benchmark_bu) == 1, "WaferConfig: Not supporting multi model prediction"

        benchmark = {f"{benchmark_name}_{self._get_config_briefing()}" : benchmark_bu.values[0]} 
        with open(benchmark_path, "w") as f:
            yaml.dump(benchmark, f)

        return benchmark_path

    def _dump_arch_config(self):
        arch_path = os.path.join(gc.database_root, "arch_bu/cerebras_like.yaml")
        with open(arch_path, 'r') as f:
            arch_config = yaml.load(f, Loader=yaml.FullLoader)

        pe_config = arch_config['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local']
        assert pe_config[-1]['name'] == "LMAC[0..3]"
        pe_config[-1]['name'] = f"LMAC[0..{self.core_mac_num-1}]"
        assert pe_config[-2]['name'] == "PEWeightRegs[0..3]"
        pe_config[-2]['name'] = f"PEWeightRegs[0..{self.core_mac_num-1}]"

        assert pe_config[0]['name'] == "PEInputBuffer"
        # FIXME:

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

        predictor = LinearProgrammingPredictor(trace_parser)
        total_latency = 0
        for layer_name in trace_parser.graph_parser.get_layers():
            total_latency += predictor.run(layer_name)

        prediction_path = os.path.join(task_root, "prediction.json")
        with open(prediction_path, "w") as f:
            json.dump(total_latency, f)

        return total_latency