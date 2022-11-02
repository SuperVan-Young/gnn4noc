import os
import time
import subprocess
import signal
import re
import dse_global_control as gc


def run_focus(benchmark_path, array_size, flit_size, mode, verbose=False, debug=False, timeout=600):
    executable = os.path.join(gc.focus_root, "focus.py")

    command = f"python {executable} -bm {benchmark_path} -d {array_size} -b 1 \
                -fr {flit_size}-{flit_size}-{flit_size} {mode}" \
                + " -debug" if debug else ""

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

def run_timeloop_mapper(layer_root, verbose=False, timeout=1):
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
    sp = subprocess.Popen(command, cwd=layer_root, shell=True, env=env, preexec_fn=os.setpgrp)

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
    if verbose: print(f"Info: Mapper search complete in {end_time - begin_time} seconds.")