import os
import random
import sys

dataset_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(dataset_root, "data")
sim_result_root = os.path.join(dataset_root, "sim_result")
assert os.path.exists(sim_result_root)

if not os.path.exists(data_root):
    os.mkdir(data_root)

sys.path.append(dataset_root)

from dgl_file_generator import DGLFileGenerator
from trace_analyzer import TraceAnalyzer

for root, dirs, files in os.walk(sim_result_root):
    if len(files) < 4:
        continue  # no out.log

    taskname = os.path.split(root)[1]
    print(taskname)
    try:
        trace_analyzer = TraceAnalyzer(taskname)
    except KeyError:
        continue
    dgl_generator = DGLFileGenerator()

    for layer in trace_analyzer.get_layers():
        dgl_generator.dump_data(trace_analyzer, layer)
