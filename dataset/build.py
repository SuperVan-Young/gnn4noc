import os
import sys
import multiprocessing as mp
import argparse

import global_control as gc

from focus_agent.multiprocess import run_focus
from focus_agent.sampler import DataSampler
from convert.trace_analyzer import TraceAnalyzer
from convert.dgl_file_generator import DGLFileGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="build")

    parser.add_argument("-f", "--focus", dest="run_focus", action="store_true",
                        help="Run FOCUS toolchain")
    parser.add_argument("-n", dest="num_samples", type=int, default=10000,
                        help="Number of samples")
    parser.add_argument("-c", "--convert", dest="convert", action="store_true",
                        help="Convert the simulation result to data. \
                        Support both runtime and afterwards conversion")
    return parser.parse_args()


def set_global_control(args):
    gc.run_focus = args.run_focus
    gc.num_samples = args.num_samples
    gc.convert = args.convert

def convert_single_data(layer_config, array_size=8, flit_size=1024):
    taskname = f"{layer_config}_b1w{flit_size}_{array_size}x{array_size}"
    task_root = os.path.join(gc.tasks_root, taskname)

    assert os.path.exists(task_root)
    try:
        trace_analyzer = TraceAnalyzer(taskname)
    except KeyError:
        return
    dgl_generator = DGLFileGenerator()
    for layer in trace_analyzer.get_layers():
        dgl_generator.dump_data(trace_analyzer, layer)


def convert_all_data():
    for root, dirs, files in os.walk(gc.tasks_root):
        taskname = os.path.split(root)[1]
        try:
            trace_analyzer = TraceAnalyzer(taskname)
        except KeyError:
            return
        dgl_generator = DGLFileGenerator()
        for layer in trace_analyzer.get_layers():
            dgl_generator.dump_data(trace_analyzer, layer)


def run_single_process(layer_config):
    ret = run_focus(layer_config)
    if gc.convert and ret == 0:
        convert_single_data(layer_config)


def run_multiple_process():
    print(f"Running {gc.num_process} processes in parallel for {gc.num_samples} samples.")

    pool = mp.Pool(processes=gc.num_process)
    data_sampler = DataSampler()

    for i in range(gc.num_samples):
        layer_config = data_sampler.get_random_sample()
        pool.apply(run_single_process, args=(layer_config, ))
    pool.close()
    pool.join()

    print(f"Building dataset complete!")


if __name__ == "__main__":
    args = parse_args()
    set_global_control(args)
    if gc.run_focus:
        run_multiple_process()
    else:
        # not running focus, just converting data
        if gc.convert:
            convert_all_data()