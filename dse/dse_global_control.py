import os
import sys

# ------------------ toolchain roots --------------------
focus_root = "/home/xuechenhao/focus_scheduler"
sys.path.append(focus_root)

database_root = os.path.join(focus_root, "database")
op_graph_root = os.path.join(focus_root, "buffer", "op_graph")
simulator_root = os.path.join(focus_root, "simulator", "tasks")
timeloop_lib_path = os.path.join(focus_root, "libs")

# let user check if the path exists
def get_op_graph_path(taskname):
    path = os.path.join(op_graph_root, f"op_graph_{taskname}.gpickle")
    return path
    
def get_outlog_path(taskname):
    path = os.path.join(simulator_root, taskname, "out.log")
    return path

def get_routing_path(taskname):
    path = os.path.join(simulator_root, taskname, "routing_board")
    return path

def get_spec_path(taskname):
    path = os.path.join(simulator_root, taskname, "spatial_spec")
    return path

# ---------------------- dse roots -----------------------
dse_root = os.path.dirname(os.path.abspath(__file__))

experiment_date = '1115_2'

task_root = os.path.join(dse_root, "tasks")
if not os.path.exists(task_root):
    os.mkdir(task_root)
task_root = os.path.join(task_root, experiment_date)
if not os.path.exists(task_root):
    os.mkdir(task_root)

fig_root = os.path.join(dse_root, "figs")
if not os.path.exists(fig_root):
    os.mkdir(fig_root)
fig_root = os.path.join(fig_root, experiment_date)
if not os.path.exists(fig_root):
    os.mkdir(fig_root)

design_points_path = os.path.join(dse_root, "design_points", "design_points_mini.list")
assert os.path.exists(design_points_path)
# --------------------- control -------------------------

num_effective_model = 2   # how many model on 1 wafer  # TODO: 2,4,8,16

timeloop_mapper_timeout = 1

multiprocess_cores = 16