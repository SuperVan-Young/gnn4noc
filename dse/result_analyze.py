import os
import numpy as np
import json
import dse_global_control as gc
import matplotlib.pyplot as plt

class ResultAnalyzer():

    def __init__(self, design_points: list) -> None:
        self.design_points = design_points

    def _cluster_design_points(self, cluster_columns: list, fixed_columns: dict):
        assert len(cluster_columns) <= 2, "ResultAnalyzer: only plot less than 3d graph!"

        colname2idx = [
            "core_buffer_size", 
            "core_buffer_bw", 
            "core_num_mac", 
            "core_noc_bw", 
            "core_noc_vc", 
            "core_noc_buffer_size", 
            "reticle_bw", 
            "core_array_h", 
            "core_array_w", 
            "wafer_mem_bw", 
            "reticle_array_h", 
            "reticle_array_w", 
        ]
        colname2idx = {k: i for i, k in enumerate(colname2idx)}

        cur_cluster = dict()
        for dp in self.design_points:
            # check fixed columns' value
            skip_dp = False
            for c, v in fixed_columns.items():
                index = dp[colname2idx[c]]
                if dp[index] != v:
                    skip_dp = True
                    break
            if skip_dp: continue

            index = tuple([dp[colname2idx[c]] for c in cluster_columns])
            if index not in cur_cluster:
                cur_cluster[index] = []
            cur_cluster[index].append(dp)
            
        return cur_cluster

    def _init_perfs(self, benchmark='gpt2-xl_tiny'):
        perfs = {k: np.inf for k in self.design_points}

        for dp in self.design_points:
            dp_ = [int(k) for k in dp]
            core_buffer_size, core_buffer_bw, core_num_mac, core_noc_bw, core_noc_vc, core_noc_buffer_size, reticle_bw, core_array_h, core_array_w, wafer_mem_bw, reticle_array_h, reticle_array_w = dp_
            config_briefs = [
                ('cm', core_num_mac),
                ('cbw', core_buffer_bw),
                ('csz', core_buffer_size),
                ('nvc', core_noc_vc),
                ('nbw', core_noc_bw),
                ('nsz', core_noc_buffer_size),
                ('nah', core_array_h),
                ('naw', core_array_w),
                ('rbw', reticle_bw),
                ('rah', reticle_array_h),
                ('raw', reticle_array_w),
                ('wmbw', wafer_mem_bw),
            ]
            briefing = benchmark + "_" + "_".join([f"{k}{v}" for k, v in config_briefs])

            try:
                with open(os.path.join(gc.dse_root, "tasks", briefing, "prediction.json"), 'r') as f:
                    a = json.load(f)
                perfs[dp] = int(a)
            except:
                continue

        return perfs

    def plot_cluster(self, cluster_columns: list, fixed_columns: dict, benchmark='gpt2-xl_tiny', agg='min'):
        cluster = self._cluster_design_points(cluster_columns, fixed_columns)
        perfs = self._init_perfs(benchmark)
        index_to_agg_perf = dict()

        for index, cur_cluster in cluster.items():
            cur_perfs = [perfs[c] for c in cur_cluster]
            if agg == 'min':
                cur_agg_perf = np.min(cur_perfs)
            elif agg == 'max':
                cur_agg_perf = np.max(cur_perfs)
            elif agg == 'mean':
                cur_agg_perf = np.mean(cur_perfs)
            else:
                raise RuntimeError(f"Invalid agg {agg}")
            index_to_agg_perf[index] = cur_agg_perf

        dim = len(cluster_columns)
        if dim == 1:
            x = list(index_to_agg_perf.keys())
            y = list(index_to_agg_perf.values())

            plt.plot(x, y, marker='x')
            plt.xlabel(cluster_columns[0])
            plt.ylabel('total latency')

            fig_title = benchmark + "_" + "_".join(cluster_columns)
            fig_title += "_" + "_".join([f"{k}{v}" for k, v in fixed_columns])
            fig_title += f"_agg_{agg}"
            plt.title(fig_title)

            fig_path = os.path.join(gc.dse_root, "figs", f"{fig_title}.png")

            plt.savefig(fig_path)
            plt.clf()

        elif dim == 2:
            raise NotImplementedError
        else:
            raise RuntimeError(f"Invalid dim {dim}")


if __name__ == "__main__":
    design_points = []
    with open("design_points.list", 'r') as f:
        for line in f:
            l = line.strip('[]\n').split(',')
            l = [float(s) for s in l]
            design_points.append(tuple(l))

    analyzer = ResultAnalyzer(design_points)
 
    cluster_columns = ["core_noc_buffer_size"]
    fixed_columns = dict()
    analyzer.plot_cluster(cluster_columns, fixed_columns, agg='min')
    analyzer.plot_cluster(cluster_columns, fixed_columns, agg='max')
    analyzer.plot_cluster(cluster_columns, fixed_columns, agg='mean')