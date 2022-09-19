import os
import sys
sys.path.append("..")

import yaml
import subprocess
import time

import global_control as gc

class FocusAgent():
    """Provide support on running focus and fetching results.
    Focus toolchain should support running fake traces.
    """
    
    def __init__(self, fake_trace: bool, simulate: bool) -> None:
        self.fake_trace = fake_trace
        self.simulate = simulate

    def run_focus(self, benchmark_path: str, array_size: int, flit_size: int, timeout=300, verbose=False):
        """Run focus with timeout control.
        Raises: TimeoutError
        """
        focus_path = os.path.join(gc.focus_root, "focus.py")
        args = "d"
        args = (args + "s") if self.simulate else args
        args = (args + "g") if self.fake_trace else args
        command = f"python {focus_path} -bm {benchmark_path} -d {array_size} -b 1 \
                    -fr {flit_size}-{flit_size}-{flit_size} {args}"

        begin_time = time.time()
        if verbose:
            sp = subprocess.Popen(command, shell=True, preexec_fn=os.setpgrp)
        else:
            sp = subprocess.Popen(command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    shell=True, preexec_fn=os.setpgrp)
        try:
            sp.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print("Info: running FOCUS timeout.")
            raise TimeoutError
        except:
            raise RuntimeError            
            
        end_time = time.time()
        print(f"Info: running FOCUS complete in {end_time - begin_time} seconds.")

    def get_op_graph_path(self, taskname):
        path = os.path.join(gc.op_graph_root, f"op_graph_{taskname}.gpickle")
        return path if os.path.exists(path) else None
    
    def get_outlog_path(self, taskname):
        path = os.path.join(gc.simulator_root, taskname, "out.log")
        return path if os.path.exists(path) else None

    def get_routing_path(self, taskname):
        path = os.path.join(gc.simulator_root, taskname, "routing_board")
        return path if os.path.exists(path) else None

    def get_spec_path(self, taskname):
        path = os.path.join(gc.simulator_root, taskname, "spatial_spec")
        return path if os.path.exists(path) else None