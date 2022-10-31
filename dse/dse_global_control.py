import os
import sys

# ------------------ toolchain roots --------------------
focus_root = "/home/xuechenhao/focus_scheduler"
sys.path.append(focus_root)

database_root = os.path.join(focus_root, "database")
op_graph_root = os.path.join(focus_root, "buffer", "op_graph")
simulator_root = os.path.join(focus_root, "simulator", "tasks")

def get_op_graph_path(taskname):
    path = os.path.join(op_graph_root, f"op_graph_{taskname}.gpickle")
    return path if os.path.exists(path) else None
    
def get_outlog_path(taskname):
    path = os.path.join(simulator_root, taskname, "out.log")
    return path if os.path.exists(path) else None

def get_routing_path(taskname):
    path = os.path.join(simulator_root, taskname, "routing_board")
    return path if os.path.exists(path) else None

def get_spec_path(taskname):
    path = os.path.join(simulator_root, taskname, "spatial_spec")
    return path if os.path.exists(path) else None

# ---------------------- dse roots -----------------------
dse_root = os.path.dirname(os.path.abspath(__file__))
task_root = os.path.join(dse_root, "tasks1030")
fig_root = os.path.join(dse_root, "figs1030")