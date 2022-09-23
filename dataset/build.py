from io import open_code
import os
import sys
import multiprocessing as mp
import argparse

import global_control as gc

from focus_agent.sampler import LayerSampler
from focus_agent.focus_agent import FocusAgent
from trace_parser.trace_parser import TraceParser
from graph_generator.hyper import HyperGraphGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="build")

    parser.add_argument("-f", "--focus", dest="run_focus", action="store_true",
                        help="Run FOCUS toolchain and fetch results")
    parser.add_argument("-n", dest="num_samples", type=int, default=10000,
                        help="Number of samples")
    parser.add_argument("-c", "--convert", dest="convert", action="store_true",
                        help="Convert the simulation result to data.")
    parser.add_argument("-w", "--worker", type=int, default=4,
                        help="Number of working threads.")
    parser.add_argument("--array_size", type=int, default=8,
                        help="core array size")
    parser.add_argument("--flit_size", type=int, default=1024,
                        help="flit size (actually useless for performance)")
    return parser.parse_args()

def set_global_control(args):
    gc.build_run_focus = args.run_focus
    gc.build_num_samples = args.num_samples
    gc.build_convert = args.convert
    gc.build_worker = args.worker
    gc.build_array_size = args.array_size
    gc.build_flit_size = args.flit_size

def build():
    if gc.build_run_focus:
        print("Build: running FOCUS ...")

        sampler = LayerSampler()
        layers = [sampler.get_random_sample() for i in range(gc.build_num_samples)]

        with mp.Pool(processes=gc.build_num_process) as pool:
            pool.map(run_focus, layers)

        print("Build: running FOCUS complete!")

    if gc.build_convert:
        print("Build: converting data ...")

        convert_data()

        print("Build: converting data complete!")


def run_focus(layer):
    """Dump model, run simulation, fetch traces, clean up if fails.
    """
    taskname = str(layer) + f"_b1w{gc.build_flit_size}_{gc.build_array_size}x{gc.build_array_size}"
    task_root = os.path.join(gc.tasks_root, taskname)
    if not os.path.exists(task_root):
        os.mkdir(task_root)
        
    layer.dump(task_root, "model")

    task_root = os.path.join(gc.tasks_root, taskname)
    benchmark_path =  os.path.join(task_root, "model.yaml")
    agent = FocusAgent(fake_trace=True, simulate=True)
    
    try:
        agent.run_focus(benchmark_path, gc.build_array_size, gc.build_flit_size, timeout=300)
    except TimeoutError:
        print(f"Build: simulating layer {taskname} timeout.")
        os.system(f"rm -r {task_root}")
        return

    graph_path = agent.get_op_graph_path(taskname)
    assert graph_path != None
    os.system(f"cp {graph_path} {task_root}/op_graph.gpickle")
    outlog_path = agent.get_outlog_path(taskname)
    assert outlog_path != None
    os.system(f"cp {outlog_path} {task_root}")
    routing_path = agent.get_routing_path(taskname)
    assert routing_path != None
    os.system(f"cp {routing_path} {task_root}")
    spec_path = agent.get_spec_path(taskname)
    assert spec_path != None
    os.system(f"cp {spec_path} {task_root}")
    
    print(f"Build: simulating layer {taskname} successfully!")

def convert_data():
    tasknames = next(os.walk(gc.tasks_root))[1]

    for taskname in tasknames:
        root = os.path.join(gc.tasks_root, taskname)

        try:
            files = ['op_graph.gpickle', 'out.log', 'routing_board', 'spatial_spec']
            for file in files:
                assert os.path.exists(os.path.join(root, file))
        except:
            print(f"Build: missing data when converting {taskname}.")
            os.system(f"rm -r {root}")
            continue
        
        print(f"Converting {taskname}")

        graph_path = os.path.join(root, 'op_graph.gpickle')
        outlog_path = os.path.join(root, 'out.log')
        routing_path = os.path.join(root, 'routing_board')
        spec_path = os.path.join(root, 'spatial_spec')
        trace_parser = TraceParser(graph_path, outlog_path, routing_path, spec_path)

        try:
            graph_generator = HyperGraphGenerator(trace_parser, predict=False)
            graph = graph_generator.generate_graph()
            label = graph_generator.generate_label()
            savepath = os.path.join(gc.data_root, f"{taskname}.pkl")
            graph_generator.save_data(savepath, graph, label)
        except:
            print(f"Info: Error in converting {taskname}")


if __name__ == "__main__":
    args = parse_args()
    set_global_control(args)
    build()