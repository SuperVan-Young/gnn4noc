import os
import sys
import numpy as np
sys.path.append("..")

from exp_predictor import ExperiencePredictor
from lp_predictor import LinearProgrammingPredictor
from trace_parser.trace_parser import TraceParser
from focus_agent.focus_agent import FocusAgent

real_models = [
    "alexnet",
    "bert",
    "bert-large",
    "flappybird",
    "inception",
    "mnasnet",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "resnet50",
    "resnext50_32x4d",
    "ssd_r34",
    "unet",
    "vgg16",
    "wide_resnet50_2",
]

def predict_real_model(model_name):
    
    agent = FocusAgent(False, False)
    trace_parser = TraceParser(
        agent.get_op_graph_path(model_name),
        agent.get_outlog_path(model_name),
        agent.get_routing_path(model_name),
        agent.get_spec_path(model_name),
    )
    # predictor = ExperiencePredictor(trace_parser)
    predictor = ExperiencePredictor(trace_parser)
    total_latency = 0
    for layer_name in trace_parser.graph_parser.get_layers():
        # total_latency += predictor.predict_latency(layer_name)
        total_latency += predictor.predict_latency(layer_name)
    
    ground_truth = trace_parser.outlog_parser.get_total_latency()
    print(model_name)
    print(f"prediction: {total_latency}")
    print(f"ground truth: {ground_truth}") 
    print()

    return total_latency, ground_truth

if __name__ == "__main__":
    # focus need to checkout to gnn branch
    # git accidentally added op_graph in that branch

    latencies = []  # (estimation, ground truth)

    for model in real_models:
        for i in [4, 6, 8]:
            model_name = f"{model}_{i}_b1w1024_8x8"
            try:
                latency = predict_real_model(model_name)
                latencies.append(latency)
            except:
                print(f"error in {model_name}")
                continue

    print(f"summary")
    errors = np.abs([(x - y) / y for x, y in latencies])
    print(f"mean = {np.average(errors)}")
    print(f"std = {np.std(errors)}")