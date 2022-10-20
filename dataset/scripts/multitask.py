import sys
sys.path.append("..")

import os
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import time
import yaml
import random

import global_control as gc
from focus_agent.sampler import LayerSampler, LayerSample
from focus_agent.focus_agent import FocusAgent
from trace_parser.trace_parser import TraceParser
from graph_generator.hyper import HyperGraphGenerator
from trace_parser.visualizer import OpGraphVisualizer
from predictor.exp_predictor import ExperiencePredictor
from predictor.lp_predictor import LinearProgrammingPredictor

def build_multitask_model(num_layer, model_tag):
    """Returns: a new taskname
    """
    sampler = LayerSampler()
    models = {}
    taskname = []
    for i in range(num_layer):
        sample = sampler.get_random_sample()
        timetag = random.randint(0, 1 << 20)
        model_name = f"model_{timetag}"  # easy to recognize ...
        layer_name = str(sample) + f"_tag{timetag}_layer1"
        models[model_name] = [{layer_name: 2}]
        taskname.append(model_name)
    taskname = "_".join(taskname)
    task_root = os.path.join("multitasks", taskname)
    os.mkdir(task_root)
    benchmark_path = os.path.join(task_root, "model.yaml")
    with open(benchmark_path, 'w') as f:
        yaml.dump(models, f)

    # run focus 
    agent = FocusAgent(True, True)
    try:
        agent.run_focus(benchmark_path, 8, 1024, timeout=300, verbose=False)
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
    except:
        # print(f"Simulation timeout")
        os.system(f"rm -r {task_root}")
        return None  # very likely to happen

    return taskname


def predict_multitask_latency(taskname):
    if taskname is None:
        return None

    agent = FocusAgent(True, True)
    taskname = taskname + f"_b1w1024_8x8"
    # why cannot find trace files when taskname is none ???!!!
    try:
        trace_parser = TraceParser(
            agent.get_op_graph_path(taskname),
            agent.get_outlog_path(taskname),
            agent.get_routing_path(taskname),
            agent.get_spec_path(taskname),
        )
    except:
        return None

    predictor = LinearProgrammingPredictor(trace_parser)
    layer_names = trace_parser.graph_parser.get_layers()
    latencies = dict()
    for layer in layer_names:
        latencies[layer] = predictor.run(layer)
    ground_truth = trace_parser.outlog_parser.get_total_latency()
    latencies['total'] = ground_truth
    print(latencies)

    return latencies

if __name__ == '__main__':
    for i in range(300):
        taskname = build_multitask_model(2, i)
        predict_multitask_latency(taskname)