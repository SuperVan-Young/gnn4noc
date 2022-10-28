import os
from wafer_config import WaferConfig
from multiprocessing import Pool
import json

def run_single_design_point(design_point, run_timeloop=False):
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
    config.run_focus('gpt2-xl_tiny', run_timeloop=run_timeloop, verbose=True)

class WaferSearchSpace():

    def __init__(self, design_points):
        self.total_design_points = len(design_points)
        self.design_point_cluster = self._cluster_arch_config(design_points)

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
        print(f"number of clusters: {len(self.design_point_cluster)}")

        for key, cluster in self.design_point_cluster.items():
            # warm up
            warm_up = cluster[0]
            run_single_design_point(warm_up, run_timeloop=True)
            with Pool(processes=8) as pool:
                pool.map(run_single_design_point, cluster)
            
            print(f"Search space: cluster {key} complete!")

if __name__ == "__main__":
    design_points = []
    with open("design_points.list", 'r') as f:
        for line in f:
            l = line.strip('[]\n').split(',')
            l = [float(s) for s in l]
            design_points.append(l)
    search_space = WaferSearchSpace(design_points)
    search_space.run()