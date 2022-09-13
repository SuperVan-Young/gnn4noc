import os
import sys
sys.path.append("..")

import yaml
import subprocess
import signal
import time

import global_control as gc

def run_focus(layer_config, array_size=8, flit_size=1024, timeout=300):
    print(f"Info: Worker process running sample {layer_config}")
    taskname = f"{layer_config}_b1w{flit_size}_{array_size}x{array_size}"
    task_root = os.path.join(gc.tasks_root, taskname)
    simulator_task_root = os.path.join(gc.simulator_root, taskname)

    if not os.path.exists(task_root):
        os.mkdir(task_root)

    # dump model config
    data = {layer_config: [{f"{layer_config}": 2}]} # number is neglected 
    model_path = os.path.join(task_root, "model.yaml")
    with open(model_path, "w") as f:
        yaml.dump(data, f)

    # run focus toolchain
    focus_path = os.path.join(gc.focus_root, "focus.py")
    command = f"python {focus_path} -bm {model_path} -d {array_size} -b 1 -fr {flit_size}-{flit_size}-{flit_size} gsd"

    begin_time = time.time()
    sp = subprocess.Popen(command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            shell=True, preexec_fn=os.setpgrp)
    try:
        sp.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("Info: FOCUS timeout. Dumping the sample.")
        os.system(f"rm -r {task_root}")
        return -1
        
    end_time = time.time()
    print(f"Info: FOCUS generated a valid sample in {end_time - begin_time} seconds.")

    # otherwise, fetch the result to task root
    op_graph_path = f"op_graph_{taskname}.gpickle"
    os.system(f"cp {gc.op_graph_root}/{op_graph_path} {task_root}/op_graph.gpickle")
    for file in ['out.log', 'routing_board', 'spatial_spec']:
        os.system(f"cp {simulator_task_root}/{file} {task_root}/{file}")

    return 0

    