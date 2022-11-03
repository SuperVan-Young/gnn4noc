import os
from wafer_config import WaferConfig
from multiprocessing import Pool
import json
import yaml
import dse_global_control as gc
from wafer_config import run_focus

benchmark_name = 'dall-e-128'

def parse_design_point_list(list_path):
    design_points = []
    with open(list_path, 'r') as f:
        for line in f:
            l = line.strip('[]\n').split(',')
            l = [float(s) for s in l]
            design_points.append(l)
    return design_points

def run_single_design_point(design_point):
    design_point = [int(s) for s in design_point]
    core_buffer_size, core_buffer_bw, core_num_mac, core_noc_bw, core_noc_vc, core_noc_buffer_size, reticle_bw, core_array_h, core_array_w, wafer_mem_bw, reticle_array_h, reticle_array_w = design_point
    config = WaferConfig(
        core_num_mac = core_num_mac, 
        core_buffer_bw = core_buffer_bw, 
        core_buffer_size = core_buffer_size, 
        core_noc_bw = core_noc_bw, 
        core_noc_vc = core_noc_vc, 
        core_noc_buffer_size = core_noc_buffer_size, 
        core_array_h = core_array_h, 
        core_array_w = core_array_w, 
        reticle_bw = reticle_bw, 
        reticle_array_h = reticle_array_h, 
        reticle_array_w = reticle_array_w, 
        wafer_mem_bw = wafer_mem_bw, 
    )
    try:
        config.run(run_timeloop=False)
    except:
        print(f"Error: {design_point}")
        return
    print(f"Success: {design_point}")

def warm_up(cluster):
    """Gather all layer scaling factors that need to run timeloop
    All handwritten, really ugly, but save time!
    """
    scaling_factors = set()

    for dp in cluster:
        dp = [int(s) for s in dp]
        core_buffer_size, core_buffer_bw, core_num_mac, core_noc_bw, core_noc_vc, core_noc_buffer_size, reticle_bw, core_array_h, core_array_w, wafer_mem_bw, reticle_array_h, reticle_array_w = dp
        config = WaferConfig(
            core_num_mac = core_num_mac, 
            core_buffer_bw = core_buffer_bw, 
            core_buffer_size = core_buffer_size, 
            core_noc_bw = core_noc_bw, 
            core_noc_vc = core_noc_vc, 
            core_noc_buffer_size = core_noc_buffer_size, 
            core_array_h = core_array_h, 
            core_array_w = core_array_w, 
            reticle_bw = reticle_bw, 
            reticle_array_h = reticle_array_h, 
            reticle_array_w = reticle_array_w, 
            wafer_mem_bw = wafer_mem_bw, 
        )
        scaling_factors = scaling_factors.union(config._get_layer_scaling_factor(benchmark_name))

    print(f"scaling factors: {scaling_factors}")

    benchmark_bu_path = os.path.join(gc.dse_root, "benchmark", f"{benchmark_name}.yaml")
    with open(benchmark_bu_path, 'r') as f:
        benchmark_bu = yaml.load(f, Loader=yaml.FullLoader)
    benchmark_tmp_path = os.path.join(gc.dse_root, "tmp_benchmark.yaml")

    for factor in scaling_factors:
        tmp_benchmark = [{list(l.keys())[0]: list(l.values())[0] * factor} for l in benchmark_bu[benchmark_name]]
        tmp_benchmark = {'tmp-model': tmp_benchmark}
        with open(benchmark_tmp_path, 'w') as f:
            yaml.dump(tmp_benchmark, f)

        run_focus(benchmark_tmp_path, 64, 1024, 'ted', verbose=True, timeout=1200)

class WaferSearchSpace():

    def __init__(self, design_points):
        self.total_design_points = len(design_points)
        self.design_points = design_points
        # self.design_point_cluster = self._cluster_arch_config(design_points)

    def _cluster_arch_config(self, design_points):
        cluster = dict()
        for dp in design_points:
            index = (dp[0], dp[1], dp[2]) # buffer size, buffer bw, num mac
            if index not in cluster.keys():
                cluster[index] = []
            cluster[index].append(dp)
        return cluster

    def run(self):
        print(f"total design points: {self.total_design_points}")
        # print(f"number of clusters: {len(self.design_point_cluster)}")

        for dp in self.design_points:
            print(dp)
            run_single_design_point(dp)
            
if __name__ == "__main__":
    list_path = os.path.join(gc.dse_root, "design_points/design_points_203.list")
    design_points = parse_design_point_list(list_path)
    search_space = WaferSearchSpace(design_points)
    search_space.run()