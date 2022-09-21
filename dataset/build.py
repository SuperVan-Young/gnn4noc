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
        pool = mp.Pool(gc.build_num_process)
        for i in range(gc.build_num_samples):
            layer = sampler.get_random_sample()
            taskname = str(layer) + f"_b1w{gc.build_flit_size}_{gc.build_array_size}x{gc.build_array_size}"
            task_root = os.path.join(gc.tasks_root, taskname)
            if not os.path.exists(task_root):
                os.mkdir(task_root)
                
            layer.dump(task_root, "model")
            pool.apply_async(run_focus, args=(taskname, i, ))
        
        pool.close()
        pool.join()

        print("Build: running FOCUS complete!")

    if gc.build_convert:
        print("Build: converting data ...")

        convert_data()

        print("Build: converting data complete!")


def run_focus(taskname, num):
    """Run simulation, fetch traces, clean up if fails.
    """
    task_root = os.path.join(gc.tasks_root, taskname)
    benchmark_path =  os.path.join(task_root, "model.yaml")
    agent = FocusAgent(fake_trace=True, simulate=True)
    
    try:
        agent.run_focus(benchmark_path, gc.build_array_size, gc.build_flit_size)

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
    
    except TimeoutError:
        print(f"Build: simulating {num}-th layer {taskname} timeout.")
        os.system(f"rm -r {task_root}")
        return    

    except:
        print(f"Build: simulating {num}-th layer {taskname} failed.")
        os.system(f"rm -r {task_root}")
        return    
    
    print(f"Build: simulating {num}-th layer {taskname} successfully!")

def convert_data():
    for root, dirs, files in os.walk(gc.tasks_root):
        if root == gc.tasks_root:
            continue
        taskname = os.path.split(root)[1]

        try:
            assert 'op_graph.gpickle' in files
            assert 'out.log' in files
            assert 'routing_board' in files
            assert 'spatial_spec' in files
        except:
            print("Build: warning, missing data during conversion.")
            continue

        graph_path = os.path.join(root, 'op_graph.gpickle')
        outlog_path = os.path.join(root, 'out.log')
        routing_path = os.path.join(root, 'routing_board')
        spec_path = os.path.join(root, 'spatial_spec')
        trace_parser = TraceParser(graph_path, outlog_path, routing_path, spec_path)

        graph_generator = HyperGraphGenerator(trace_parser, predict=True)
        graph = graph_generator.generate_graph()
        label = graph_generator.generate_label()
        savepath = os.path.join(gc.data_root, f"{taskname}.pkl")
        graph_generator.save_data()



if __name__ == "__main__":
    args = parse_args()
    set_global_control(args)
    build()