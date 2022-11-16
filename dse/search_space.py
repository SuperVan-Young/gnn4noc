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

def parse_design_point(dp):
    core_buffer_size, core_buffer_bw, core_num_mac, core_noc_bw, core_noc_vc, core_noc_buffer_size, reticle_bw, core_array_h, core_array_w, wafer_mem_bw, reticle_array_h, reticle_array_w = [int(s) for s in dp]
    return {
        "core_num_mac":  core_num_mac, 
        "core_buffer_bw":  core_buffer_bw, 
        "core_buffer_size":  core_buffer_size, 
        "core_noc_bw":  core_noc_bw, 
        "core_noc_vc":  core_noc_vc, 
        "core_noc_buffer_size":  core_noc_buffer_size, 
        "core_array_h":  core_array_h, 
        "core_array_w":  core_array_w, 
        "reticle_bw":  reticle_bw, 
        "reticle_array_h":  reticle_array_h, 
        "reticle_array_w":  reticle_array_w, 
        "wafer_mem_bw":  wafer_mem_bw, 
    }

def run_dump_config_spec(dp):
    config = WaferConfig(**parse_design_point(dp))
    config.run(dump_benchmark=True, invoke_timeloop_mapper=False, invoke_timeloop_model=False, invoke_focus=False, predict=False)
    return

def run_config_with_invoke_focus(config, verbose=True, debug=True):
    try:
        config.run(dump_benchmark=False, invoke_timeloop_mapper=False, invoke_timeloop_model=False, invoke_focus=True, predict=True)
    except:
        if verbose: print(f"Error: predictor {config._get_config_briefing()}")
        if debug:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=sys.stderr)
            return
    if verbose: print(f"Success: predictor {config._get_config_briefing()}")

def run_config(config, verbose=True, debug=True):
    try:
        config.run(dump_benchmark=False, invoke_timeloop_mapper=False, invoke_timeloop_model=False, invoke_focus=False, predict=True)
    except:
        if verbose: print(f"Error: predictor {config._get_config_briefing()}")
        if debug:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=sys.stderr)
            return
    if verbose: print(f"Success: predictor {config._get_config_briefing()}")

class WaferSearchSpace():

    def __init__(self, design_points, ):
        self.total_design_points = len(design_points)
        self.design_points = design_points

    def run(self, dump_config_spec, invoke_timeloop_mapper, invoke_timeloop_model, invoke_focus, predict, verbose=False, debug=False):
        print(f"total design points: {self.total_design_points}")

        if dump_config_spec:
            with Pool(processes=gc.multiprocess_cores) as pool:
                pool.map(run_dump_config_spec, self.design_points)
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
            with Pool(processes=gc.multiprocess_cores) as pool:
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
            with Pool(processes=gc.multiprocess_cores) as pool:
                pool.map(run_timeloop_model, layer_roots)

        if predict:
            dp_predict = []
            for dp in self.design_points:
                config = WaferConfig(**parse_design_point(dp))
                layer_root = config._get_config_briefing()
                prediction_root = os.path.join(gc.task_root, layer_root, "prediction")
                if not os.path.exists(prediction_root):
                    dp_predict.append(config)
                    continue
                is_append_dp = True
                for _, __, files in os.walk(prediction_root):
                    if len(files) == 2:
                        is_append_dp = False
                        break
                # rerun or not
                # if is_append_dp: dp_predict.append(config)
                dp_predict.append(config)
            
            print(f"Predict: {len(dp_predict)}")

            # for dp in dp_predict:
            #     run_config(dp)
            with Pool(processes=gc.multiprocess_cores) as pool:
                if invoke_focus:
                    pool.map(run_config_with_invoke_focus, dp_predict)
                else:
                    pool.map(run_config, dp_predict)

            
if __name__ == "__main__":
    design_points = parse_design_point_list(gc.design_points_path)
    search_space = WaferSearchSpace(design_points)
    search_space.run(
        dump_config_spec=False, 
        invoke_timeloop_mapper=False, 
        invoke_timeloop_model=False, 
        invoke_focus=True, 
        predict=True, 
        verbose=True, debug=True) 