import os
import numpy as np
import json
import dse_global_control as gc
import matplotlib.pyplot as plt
import matplotlib
from search_space import parse_design_point_list
import traceback

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
                v_ = int(dp[colname2idx[c]])
                if v_ != v:
                    skip_dp = True
                    break
            if skip_dp: continue

            index = tuple([dp[colname2idx[c]] for c in cluster_columns])
            if index not in cur_cluster:
                cur_cluster[index] = []
            cur_cluster[index].append(dp)
            
        return cur_cluster

    def _init_perfs(self, benchmark='gpt2-xl-tiny'):
        perfs = {k: dict() for k in self.design_points}  # dp: {layer: latency}

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
            briefing = "_".join([f"{k}{v}" for k, v in config_briefs])
            task_root = briefing

            benchmark_root = os.path.join(gc.task_root, task_root, "benchmark")
            assert os.path.exists(benchmark_root), f"{benchmark_root} not exists!"
            for _, _, files in os.walk(benchmark_root):
                for file in files:
                    if benchmark not in file:  continue
                    try:
                        cur_benchmark_name = file[:-5] + "_" + briefing + ".json"
                        prediction_path = os.path.join(gc.task_root, task_root, "prediction", cur_benchmark_name)
                        assert os.path.exists(prediction_path), f"{prediction_path} does not exists!"
                        with open(prediction_path, 'r') as f:
                            a = json.load(f)
                            perfs[dp]['latency'] = sum([v for v in a['prediction'].values()])
                            perfs[dp]['ratio'] = sum([v for v in a['prediction'].values()]) / sum([v for v in a['theoretical'].values()])
                    except:
                        print(f"Warning: Missing files in loading prediction result of {dp}, automatically filling with inf")
                        traceback.print_exc()
                        perfs[dp]['latency'] = np.inf
                        perfs[dp]['ratio'] = np.inf
                break

        return perfs

    def plot_cluster(self, cluster_columns: list, fixed_columns: dict, benchmark='gpt2-xl-tiny', agg='min'):
        COMP_BOUND = 1.2  # compuration bound

        dp_clusters = self._cluster_design_points(cluster_columns, fixed_columns)  # cluster_cols -> dps
        perfs = self._init_perfs(benchmark)  # dp -> perf_dict
        index_to_best_dp = dict()  # cluster_cols -> best dp

        # for each cluster, find the best dp according to some metric
        def get_better_dp(a, b):
            return a if perfs[a]['latency'] <= perfs[b]['latency'] else b

        for index, dps in dp_clusters.items():
            best_dp = None
            for dp in dps:
                if best_dp is None:
                    best_dp = dp
                    continue
                best_dp = get_better_dp(best_dp, dp)
            index_to_best_dp[index] = best_dp

        dim = len(cluster_columns)
        if dim == 1:
            x = [k for k, v in index_to_best_dp.items() if perfs[v]['ratio'] != np.inf]
            y = [perfs[v]['latency'] for v in index_to_best_dp.values() if perfs[v]['ratio'] != np.inf]
            plt.plot(x, y)

            c = np.array([perfs[v]['ratio'] for v, v in index_to_best_dp.items() if perfs[v]['ratio'] != np.inf])
            c = np.log(c)
            cmap = matplotlib.cm.get_cmap('bone')
            norm = matplotlib.colors.Normalize(c.min(), c.max())
            c = cmap(norm(c.tolist()))
            plt.scatter(x, y, marker='x', color=c)
            plt.xlabel(cluster_columns[0])
            plt.ylabel('total latency')

        elif dim == 2:
            ax = plt.axes(projection='3d')

            x = np.log2([k[0] for k, v in index_to_best_dp.items() if perfs[v]['ratio'] != np.inf])
            y = np.log2([k[1] for k, v in index_to_best_dp.items() if perfs[v]['ratio'] != np.inf])
            z = np.array([perfs[v]['latency'] for k, v in index_to_best_dp.items() if perfs[v]['ratio'] != np.inf])
            c = np.array([perfs[v]['ratio'] for k, v in index_to_best_dp.items() if perfs[v]['ratio'] != np.inf])
            c = np.log(c)
            cmap = matplotlib.cm.get_cmap('bone')
            norm = matplotlib.colors.Normalize(c.min(), c.max())
            c = cmap(norm(c.tolist()))
            if len(z):
                ax.bar3d(x, y, 0, 0.25, 0.25, z, color=c)

            ax.view_init(elev=33, azim=66)
            ax.set_xlabel(f"{cluster_columns[0]} (log)")
            ax.set_ylabel(f"{cluster_columns[1]} (log)")
            plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm))

        else:
            raise RuntimeError(f"Invalid dim {dim}")

        fig_title = benchmark + "_" + "_".join(cluster_columns)
        fig_title += "_" + "_".join([f"{k}{v}" for k, v in fixed_columns.items()])
        fig_title += f"_agg_{agg}"
        plt.title(fig_title)

        fig_path = os.path.join(gc.fig_root, f"{fig_title}.png")

        plt.savefig(fig_path)
        plt.clf()


if __name__ == "__main__":
    design_points = parse_design_point_list(os.path.join(gc.dse_root, "design_points/design_points_2.list"))
    design_points = [tuple(i) for i in design_points]
    analyzer = ResultAnalyzer(design_points)
    
    # single variable, other variable choose the best setting
    for prop in ["core_num_mac", "core_buffer_bw", "core_buffer_size", "core_noc_bw"]:
        cluster_columns = [prop]
        fixed_columns = dict()
        try:
            analyzer.plot_cluster(cluster_columns, fixed_columns, agg='min', benchmark="gpt2-xl")
            analyzer.plot_cluster(cluster_columns, fixed_columns, agg='min', benchmark="dall-e-128")
        except:
            print(f"error: {cluster_columns} {fixed_columns}")
            # traceback.print_exc()
            continue

    # for fixed buffer size, perf <- mac and noc?
    for core_buffer_size in 2 ** np.arange(5, 12):
        cluster_columns = [
            'core_num_mac',
            'core_noc_bw',
        ]
        fixed_columns = {
            'core_buffer_size': core_buffer_size,
            # 'core_buffer_bw': core_buffer_bw,  # fix this equals fixing num mac ...
        }
        try:
            analyzer.plot_cluster(cluster_columns, fixed_columns, agg='min', benchmark='gpt2-xl')
            analyzer.plot_cluster(cluster_columns, fixed_columns, agg='min', benchmark='dall-e-128')
        except:
            print(f"error: {cluster_columns} {fixed_columns}")
            traceback.print_exc()
            continue

    # for fixed buffer size, perf <- mac and noc?
    for core_noc_bw in 2 ** np.arange(5, 12):
        for core_buffer_size in 2 ** np.arange(5, 12):
            cluster_columns = [
                'core_num_mac',
            ]
            fixed_columns = {
                'core_noc_bw': core_noc_bw, 
                'core_buffer_size': core_buffer_size,
                # 'core_buffer_bw': core_buffer_bw,  # fix this equals fixing num mac ...
            }
            try:
                analyzer.plot_cluster(cluster_columns, fixed_columns, agg='min', benchmark='gpt2-xl')
                analyzer.plot_cluster(cluster_columns, fixed_columns, agg='min', benchmark='dall-e-128')
            except:
                print(f"error: {cluster_columns} {fixed_columns}")
                # traceback.print_exc()
                continue

    # for fixed noc, perf <- mac and buf?
    for core_noc_bw in 2 ** np.arange(5, 15):
        cluster_columns = [
            'core_num_mac',
            'core_buffer_size',
        ]
        fixed_columns = {
            'core_noc_bw': core_noc_bw,
            # 'core_buffer_bw': core_buffer_bw,  # fix this equals fixing num mac ...
        }
        try:
            analyzer.plot_cluster(cluster_columns, fixed_columns, agg='min', benchmark='gpt2-xl')
            analyzer.plot_cluster(cluster_columns, fixed_columns, agg='min', benchmark='dall-e-128')
        except:
            print(f"error: {cluster_columns} {fixed_columns}")
            # traceback.print_exc()
            continue