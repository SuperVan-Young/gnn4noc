import os
import time
import subprocess
import signal
import re
import dse_global_control as gc
from compiler.timeloop_agents.loop2map import Loop2Map


def run_focus(benchmark_path, array_size, flit_size, mode, timeloop_buffer_path, verbose=False, debug=False, timeout=600):
    executable = os.path.join(gc.focus_root, "focus.py")

    command = f"python {executable} -bm {benchmark_path} -d {array_size} -b 1 \
                -fr {flit_size}-{flit_size}-{flit_size} -tlb {timeloop_buffer_path} {mode}" \
                + (" -debug" if debug else "")

    begin_time = time.time()
    if verbose:
        sp = subprocess.Popen(command, shell=True, start_new_session=True)
    else:
        sp = subprocess.Popen(command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            shell=True, start_new_session=True)
    try:
        sp.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("Warning: running FOCUS timeout.")
        os.killpg(os.getpgid(sp.pid), signal.SIGTERM)
        sp.wait()

    end_time = time.time()
    print(f"Info: running FOCUS complete in {end_time - begin_time} seconds.")

def run_timeloop_mapper(layer_root, verbose=True, timeout=15):
    """ layer root has prepared:
    - top level arch spec (fetch component spec from FOCUS)
    - constraint spec
    """
    executable = os.path.join(gc.timeloop_lib_path, 'timeloop-mapper')

    arch_specs = []
    arch_specs.append(os.path.join(layer_root, "modified_arch.yaml"))
    for comp_root, dirs, files in os.walk(os.path.join(gc.database_root, "arch", "components")):
        for file in files:
            comp_spec = os.path.join(comp_root, file)
            arch_specs.append(comp_spec)
    arch_specs = " ".join(arch_specs)

    mapper_specs = os.path.join(gc.database_root, "mapper", "mapper.yaml")

    constraint_specs = os.path.join(layer_root, "constraints.yaml")

    parsed_layer_config = re.search(r"(^.*)_layer(\d+)_\d+$", os.path.split(layer_root)[1])
    model_name, layer_id = parsed_layer_config.group(1), parsed_layer_config.group(2)
    prob_specs = os.path.join(gc.database_root, model_name, f"{model_name}_layer{layer_id}.yaml")

    redirect_log =  ">" + os.path.join(layer_root, "timeloop-log.txt") + " 2>&1"

    command = " ".join([executable, arch_specs, mapper_specs, constraint_specs, prob_specs, redirect_log])
    # if verbose: print(command)

    env = os.environ.copy()
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = "{}:{}".format(os.path.join(gc.focus_root, "libs"), env["LD_LIBRARY_PATH"])
    else:
        env["LD_LIBRARY_PATH"] = "{}".format(os.path.join(gc.focus_root, "libs"))

    begin_time = time.time()

    # sp = subprocess.Popen(command, cwd=layer_root, shell=True, env=env, preexec_fn=os.setpgrp)
    sp = subprocess.Popen(command, cwd=layer_root, shell=True, env=env, start_new_session=True)
    def sigint_handler(signum, frame):
        os.killpg(os.getpgid(sp.pid), signal.SIGINT)
        exit(-1)
    signal.signal(signal.SIGINT, sigint_handler)
    
    try:
        sp.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if verbose: print("Info: Mapper timeout, stop searching now.")
        os.killpg(os.getpgid(sp.pid), signal.SIGINT)
    sp.wait()

    end_time = time.time()
    if verbose: print(f"Info: {layer_root} Mapper search complete in {end_time - begin_time} seconds.")

def run_timeloop_model(layer_root, verbose=True):
    """Run timeloop model, prepare for communication status. 
    Directly copied from FOCUS.
    """
    mapper_map_file = os.path.join(layer_root, "timeloop-mapper.map.txt")

    arch_specs = []
    arch_specs.append(os.path.join(layer_root, "modified_arch.yaml"))
    for comp_root, dirs, files in os.walk(os.path.join(gc.database_root, "arch", "components")):
        for file in files:
            comp_spec = os.path.join(comp_root, file)
            arch_specs.append(comp_spec)
    arch_specs = " ".join(arch_specs)

    parsed_layer_config = re.search(r"(^.*)_layer(\d+)_\d+$", os.path.split(layer_root)[1])
    model_name, layer_id = parsed_layer_config.group(1), parsed_layer_config.group(2)
    prob_specs = os.path.join(gc.database_root, model_name, f"{model_name}_layer{layer_id}.yaml")

    dump_mapping_file = os.path.join(layer_root, "dump_mapping.yaml")

    try:
        transformer = Loop2Map()
        transformer.transform(mapper_map_file, prob_specs, dump_mapping_file)
    except:
        if verbose: print(f"Loop2Map Error, possibly timeloop mapper map file is missing")
        return

    # invoke model for getting communication status, single process
    executable = os.path.join(gc.timeloop_lib_path, 'timeloop-model')
    command = " ".join([executable, arch_specs, dump_mapping_file, prob_specs])

    env = os.environ.copy()
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = "{}:{}".format(os.path.join(gc.focus_root, "libs"), env["LD_LIBRARY_PATH"])
    else:
        env["LD_LIBRARY_PATH"] = "{}".format(os.path.join(gc.focus_root, "libs"))

    begin_time = time.time()
    try:
        if verbose:
            model_sp = subprocess.Popen(command, cwd=layer_root, shell=True, env=env)
        else:
            model_sp = subprocess.Popen(command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        cwd=layer_root, shell=True, env=env)
    except:
        if verbose: print(f"Error in timeloop model")
    model_sp.wait()
    end_time = time.time()

    if verbose: print(f"Info: {layer_root} timeloop model finished in {end_time - begin_time} seconds")