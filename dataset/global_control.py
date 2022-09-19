import os
import sys

# ------------------ toolchain roots --------------------
focus_root = "/home/xuechenhao/focus_scheduler"
sys.path.append(focus_root)

op_graph_root = os.path.join(focus_root, "buffer", "op_graph")
simulator_root = os.path.join(focus_root, "simulator", "tasks")

# ------------------ dataset roots --------------------
dataset_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dataset_root)

tasks_root = os.path.join(dataset_root, "tasks")
if not os.path.exists(tasks_root):
    os.mkdir(tasks_root)

data_root = os.path.join(dataset_root, "data")
if not os.path.exists(data_root):
    os.mkdir(data_root)

# ------------------ Runtime configs --------------------
run_focus = False

convert = False

num_samples = 1

num_process = 16