import os
from wafer_config import WaferConfig
from multiprocessing import Pool
import dse_global_control as gc
from sp_runner import run_timeloop_mapper, run_timeloop_model
import traceback
import sys

def parse_design_point_list(list_path):
    design_points = []
    with open(list_path, 'r') as f:
        for line in f:
            l = line.strip('[]\n').split(',')
            l = [float(s) for s in l]
            design_points.append(l)
    return design_points

def dump_config_spec(dp):
    core_buffer_size, core_buffer_bw, core_num_mac, core_noc_bw, core_noc_vc, core_noc_buffer_size, reticle_bw, core_array_h, core_array_w, wafer_mem_bw, reticle_array_h, reticle_array_w = [int(s) for s in dp]
    config = WaferConfig(
        core_num_mac = core_num_mac, 
        core_buffer_bw = core_buffer_bw, 
        core_buffer_size = core_buffer_size, 
        core_noc_bw = core_noc_bw, 
        core_noc_vc = core_noc_vc, 
        core_noc_buffer_size = core_noc_buffer_size, 
        core_array_h = core_array_h, 
        core_array_w = core_array_w, 
        reticle_bw = reticle_bw, 
        reticle_array_h = reticle_array_h, 
        reticle_array_w = reticle_array_w, 
        wafer_mem_bw = wafer_mem_bw, 
    )
    config.run(invoke_timeloop_mapper=False, invoke_timeloop_model=False, predict=False)
    return

class WaferSearchSpace():

    def __init__(self, design_points, ):
        self.total_design_points = len(design_points)
        self.design_points = design_points

    def run(self, dump_config_spec, invoke_timeloop_mapper, invoke_timeloop_model, predict, verbose=False, debug=False):
        print(f"total design points: {self.total_design_points}")

        if dump_config_spec:
            with Pool(processes=32) as pool:
                pool.map(dump_config_spec, self.design_points)
            print(f"Dumping config specification complete!")

        if invoke_timeloop_mapper:
            layer_roots = []
            for _, config_dirs, __ in os.walk(gc.task_root):
                for config_dir in config_dirs:
                    for layer_root, layer_dirs, __ in os.walk(os.path.join(gc.task_root, config_dir, "layers")):
                        for layer_dir in layer_dirs:
                            tmp_root = os.path.join(layer_root, layer_dir)
                            if not os.path.exists(os.path.join(tmp_root, "timeloop-mapper.map.txt")):
                                if debug: print(f"Timeloop mapper: {tmp_root}")
                                layer_roots.append(tmp_root)
                        break
                break
            
            # some layers cannot give a valid mapping, give it up anyway
            # we say it's a design flaw
            print(f"Timeloop mapper layers: {len(layer_roots)}")
            with Pool(processes=32) as pool:
                pool.map(run_timeloop_mapper, layer_roots)

        if invoke_timeloop_model:
            layer_roots = []
            for _, config_dirs, __ in os.walk(gc.task_root):
                for config_dir in config_dirs:
                    for layer_root, layer_dirs, __ in os.walk(os.path.join(gc.task_root, config_dir, "layers")):
                        for layer_dir in layer_dirs:
                            tmp_root = os.path.join(layer_root, layer_dir)
                            if not os.path.exists(os.path.join(tmp_root, "communication.yaml")):
                                layer_roots.append(tmp_root)
                        break
                break

            print(f"Timeloop model layers: {len(layer_roots)}")
            with Pool(processes=32) as pool:
                pool.map(run_timeloop_model, layer_roots)

        if predict:
            for dp in self.design_points:
                if verbose: print(f"Running prediction for {dp}")
                core_buffer_size, core_buffer_bw, core_num_mac, core_noc_bw, core_noc_vc, core_noc_buffer_size, reticle_bw, core_array_h, core_array_w, wafer_mem_bw, reticle_array_h, reticle_array_w = [int(s) for s in dp]
                config = WaferConfig(
                    core_num_mac = core_num_mac, 
                    core_buffer_bw = core_buffer_bw, 
                    core_buffer_size = core_buffer_size, 
                    core_noc_bw = core_noc_bw, 
                    core_noc_vc = core_noc_vc, 
                    core_noc_buffer_size = core_noc_buffer_size, 
                    core_array_h = core_array_h, 
                    core_array_w = core_array_w, 
                    reticle_bw = reticle_bw, 
                    reticle_array_h = reticle_array_h, 
                    reticle_array_w = reticle_array_w, 
                    wafer_mem_bw = wafer_mem_bw, 
                )
                try:
                    config.run(invoke_timeloop_mapper=False, invoke_timeloop_model=False, predict=True)
                except:
                    if verbose: print(f"Error: predictor {dp}")
                    if debug:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=sys.stderr)
                        return
                    continue
                if verbose: print(f"Success: predictor {dp}")
            
if __name__ == "__main__":
    list_path = os.path.join(gc.dse_root, "design_points/design_points_203.list")
    design_points = parse_design_point_list(list_path)
    search_space = WaferSearchSpace(design_points, )
    search_space.run(dump_config_spec=False, invoke_timeloop_mapper=False, invoke_timeloop_model=False, predict=True, verbose=True, debug=True)