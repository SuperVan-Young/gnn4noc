import os
import sys
sys.path.append("..")

import yaml
import multiprocessing as mp

import global_control as gc

def run_focus(layer, array_size=8, flit_size=1024):
    taskname = f"{layer}_b1w{flit_size}_{array_size}x{array_size}"
    task_root = os.path.join(gc.tasks_root, taskname)
    simulator_task_root = os.path.join(gc.simulator_root, taskname)

    if not os.path.exists(task_root):
        os.mkdir(task_root)

    # dump model config
    data = {layer: [{layer: 2}]} # number is neglected 
    model_path = os.path.join(gc.tasks_root, taskname, f"model.yaml")
    with open(model_path, "w") as f:
        yaml.dump(data, f)

    # run focus toolchain
    focus_path = os.path.join(gc.focus_root, "focus.py")
    cmd_line = f"python {focus_path} -bm {model_path} -d {array_size} -b 1 -fr {flit_size}-{flit_size}-{flit_size} gsd"
    ret = os.system(cmd_line)

    # if failure happens (e.g., deadlock detected), delete this sample
    if ret != 0:
        print(f"Failed to run: {cmd_line}")
        os.removedirs(task_root)
        return 1

    # otherwise, fetch the result to task root
    else:
        op_graph_path = f"op_graph_{taskname}.gpickle"
        os.system(f"cp {gc.op_graph_root}/{op_graph_path} {task_root}/op_graph.gpickle")
        for file in ['out.log', 'routing_board', 'spatial_spec']:
            os.system(f"cp {simulator_task_root}/{file} {task_root}/{file}")
        return 0
    
    