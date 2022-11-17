import os
import numpy as np
import json
import dse_global_control as gc
import matplotlib.pyplot as plt
import matplotlib
from search_space import parse_design_point_list, parse_design_point
import traceback
import yaml
import itertools
from power import PowerAnalyzer

class ResultAnalyzer():

    def __init__(self, design_points: list, power_table: dict) -> None:
        self.design_points = design_points
        self.power_table = power_table

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

    def _init_perfs(self, benchmark, wanted_layers: list):
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

            # power analyzer
            power_analyzer = PowerAnalyzer(
                **self.power_table[str(dp_)]
            )

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
                            # select part of the layers
                            prediction = {k: v for k, v in a['prediction'].items() if k in wanted_layers}
                            computation = {k: v for k, v in a['computation'].items() if k in wanted_layers}
                            transmission = {k: max(v['wsrc']['total'], v['insrc']['total'], v['worker']['total'] / v['worker']['cnt']) for k, v in a['transmission'].items() if k in wanted_layers}

                            def get_ratio(pred, comp, trans):
                                """Return predicted latency vs. theoretical latency ratio.
                                If the task is computation bounded, return a positive log ratio.
                                Else, if the task is transmission bounded, return a negative log ratio.
                                """
                                if comp > trans:
                                    return np.log(pred / trans)
                                else:
                                    return -np.log(pred / comp)

                            ratio = {k: get_ratio(pred, computation[k], transmission[k]) for k, pred in prediction.items()}
                            perfs[dp]['latency'] = np.sum(list(prediction.values())) # total latency
                            perfs[dp]['ratio'] = np.average(list(ratio.values()))  # average latency ratio (doesn't make sense for multiple layers)
                            perfs[dp]['power'] = power_analyzer.run(a)
                    except:
                        print(f"Warning: error for {dp}, automatically filling with inf")
                        traceback.print_exc()
                        perfs[dp]['latency'] = np.inf
                        perfs[dp]['ratio'] = np.inf
                        perfs[dp]['power'] = {'total': np.inf}
                break

        return perfs

    def plot_cluster(self, cluster_columns: list, fixed_columns: dict, benchmark, wanted_layers: list):
        dp_clusters = self._cluster_design_points(cluster_columns, fixed_columns)  # cluster_cols -> dps
        perfs = self._init_perfs(benchmark, wanted_layers)  # dp -> perf_dict
        index_to_best_dp = dict()  # cluster_cols -> best dp

        # for each cluster, find the best dp according to some metric
        def get_better_dp(a, b):
            return a if perfs[a]['latency'] <= perfs[b]['latency'] else b

        def is_valid_perf(perf):
            return perf['ratio'] != np.inf

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
            x = np.log2([k[0] for k, v in index_to_best_dp.items() if is_valid_perf(perfs[v])])
            y = np.array([perfs[v]['latency'] for k, v in index_to_best_dp.items() if is_valid_perf(perfs[v])])
            c = np.array([perfs[v]['ratio'] for k, v in index_to_best_dp.items() if is_valid_perf(perfs[v])])
            cmap = matplotlib.cm.get_cmap('coolwarm')
            norm_max = max(np.abs(c.min()), np.abs(c.max()))
            norm = matplotlib.colors.Normalize(-norm_max, norm_max)
            c = cmap(norm(c.tolist()))
            plt.bar(x, y, color=c)
            plt.xlabel(f"{cluster_columns[0]} (log)")
            plt.ylabel('total latency')
            plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm))

        elif dim == 2:
            ax = plt.axes(projection='3d')

            x = np.log2([k[0] for k, v in index_to_best_dp.items() if is_valid_perf(perfs[v])])
            y = np.log2([k[1] for k, v in index_to_best_dp.items() if is_valid_perf(perfs[v])])
            z = np.array([perfs[v]['latency'] for k, v in index_to_best_dp.items() if is_valid_perf(perfs[v])])
            c = np.array([perfs[v]['ratio'] for k, v in index_to_best_dp.items() if is_valid_perf(perfs[v])])
            cmap = matplotlib.cm.get_cmap('coolwarm')
            norm_max = max(np.abs(c.min()), np.abs(c.max()))
            # norm = matplotlib.colors.Normalize(-norm_max, norm_max)
            norm = matplotlib.colors.Normalize(-6, 6)
            c = cmap(norm(c.tolist()))
            if len(z):
                ax.bar3d(x, y, 0, 0.25, 0.25, z, color=c)

            if min(z) == 0:
                for k, v in index_to_best_dp.items():
                    if perfs[v]['latency'] == 0:
                        print(k, v)

            ax.view_init(elev=33, azim=66)
            ax.set_xlabel(f"{cluster_columns[0]} (log)")
            ax.set_ylabel(f"{cluster_columns[1]} (log)")
            plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm))

        else:
            raise RuntimeError(f"Invalid dim {dim}")

        fig_title = " ".join([
            "-".join(cluster_columns),
            "-".join([f"{k}={v}" for k, v in fixed_columns.items()]),
            "-".join(wanted_layers),
        ])
        plt.title(fig_title)

        fig_path = os.path.join(gc.fig_root, f"{fig_title}.png")
        plt.savefig(fig_path)
        plt.clf()


    def plot_1d_topi(self, cluster_column, benchmark, wanted_layers:list): 
        dp_clusters = self._cluster_design_points([cluster_column], dict())  # cluster_cols -> dps
        perfs = self._init_perfs(benchmark, wanted_layers)  # dp -> perf_dict

        def latency_order(dp):
            return perfs[dp]['latency']

        def dp2text(dp):
            dp = parse_design_point(dp)
            return f"{dp['core_num_mac']}x{dp['core_noc_bw']}"
        
        cluster_labels = []
        topi_dps = []
        topi = 4
        norm_max = 0
        
        for cluster, dps in dp_clusters.items():
            assert len(cluster) == 1
            cluster_labels.append(str(cluster[0]))

            sorted_dps = sorted(dps, key=latency_order)
            assert len(sorted_dps) >= topi
            topi_dps.append(sorted_dps[:topi])

            norm_max = max(norm_max, max([np.abs(perfs[v]['ratio']) for v in sorted_dps[:topi]]))

        cmap = matplotlib.cm.get_cmap('coolwarm')
        # norm = matplotlib.colors.Normalize(-norm_max, norm_max)
        norm = matplotlib.colors.Normalize(-6, 6)

        width = 0.8
        x = np.arange(len(cluster_labels))
        for i in range(topi):
            cur_dps_perfs = [perfs[v[i]]['latency'] for v in topi_dps]
            color = [perfs[v[i]]['ratio'] for v in topi_dps]
            color = cmap(norm(color))
            labels = [v[i] for v in topi_dps]
            labels = [dp2text(dp) for dp in labels]
            p = plt.bar(x - width / 2 + width / topi * i, cur_dps_perfs, width / topi * 0.8, color=color)
            plt.bar_label(p, labels, padding=5, rotation=90.)
        
        plt.xlabel(f"{cluster_column}")
        plt.xticks(x, cluster_labels)
        plt.ylabel('total latency')

        fig_title = " ".join([
            cluster_column,
            f"top-{topi}",
            "-".join(wanted_layers),
        ])
        fig_path = os.path.join(gc.fig_root, f"{fig_title}.png")
        plt.savefig(fig_path)
        plt.clf()

    def plot_power_breakdown(self, cluster_columns, fixed_columns, benchmark):
        dp_clusters = self._cluster_design_points(cluster_columns, fixed_columns)  # cluster_cols -> dps
        perfs = self._init_perfs(benchmark, wanted_layers=[])  # dp -> perf_dict
        index_to_best_dp = dict()  # cluster_cols -> best dp
        power_labels = [
            'mac_dynamic',
            'mac_static',
            'noc_dynamic',
            'noc_static',
            'sram_dynamic',
            'sram_static',
            'inter_reticle'
        ]

        # for each cluster, find the best dp according to some metric
        def get_better_dp(a, b):
            return a if perfs[a]['power']['total'] <= perfs[b]['power']['total'] else b

        def is_valid_perf(perf):
            return perf['power']['total'] != np.inf

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
            for label in power_labels:
                x = np.log2([k[0] for k, v in index_to_best_dp.items() if is_valid_perf(perfs[v])])
                y = np.array([perfs[v]['power'][label] for k, v in index_to_best_dp.items() if is_valid_perf(perfs[v])])
                plt.bar(x, y, label=label)
                plt.xlabel(f"{cluster_columns[0]} (log)")
                plt.ylabel('power (W)')
            plt.legend()

        elif dim == 2:
            raise NotImplementedError
            ax = plt.axes(projection='3d')

            x = np.log2([k[0] for k, v in index_to_best_dp.items() if is_valid_perf(perfs[v])])
            y = np.log2([k[1] for k, v in index_to_best_dp.items() if is_valid_perf(perfs[v])])
            z = np.array([perfs[v]['latency'] for k, v in index_to_best_dp.items() if is_valid_perf(perfs[v])])
            c = np.array([perfs[v]['ratio'] for k, v in index_to_best_dp.items() if is_valid_perf(perfs[v])])
            cmap = matplotlib.cm.get_cmap('coolwarm')
            norm_max = max(np.abs(c.min()), np.abs(c.max()))
            # norm = matplotlib.colors.Normalize(-norm_max, norm_max)
            norm = matplotlib.colors.Normalize(-6, 6)
            c = cmap(norm(c.tolist()))
            if len(z):
                ax.bar3d(x, y, 0, 0.25, 0.25, z, color=c)

            if min(z) == 0:
                for k, v in index_to_best_dp.items():
                    if perfs[v]['latency'] == 0:
                        print(k, v)

            ax.view_init(elev=33, azim=66)
            ax.set_xlabel(f"{cluster_columns[0]} (log)")
            ax.set_ylabel(f"{cluster_columns[1]} (log)")
            plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm))

        else:
            raise RuntimeError(f"Invalid dim {dim}")

        fig_title = " ".join([
            "power",
            "-".join(cluster_columns),
            "-".join([f"{k}={v}" for k, v in fixed_columns.items()]),
            benchmark,
        ])
        plt.title(fig_title)

        fig_path = os.path.join(gc.fig_root, f"{fig_title}.png")
        plt.savefig(fig_path)
        plt.clf()


if __name__ == "__main__":
    design_points = parse_design_point_list(gc.design_points_path)
    design_points = [tuple(i) for i in design_points]
    with open(gc.power_table_path, 'r') as f:
        power_table = json.load(f)
    analyzer = ResultAnalyzer(design_points, power_table)

    benchmark_constraints = []
    for root, _, files in os.walk(os.path.join(gc.dse_root, 'benchmark_constraint')):
        for file in files:
            with open(os.path.join(root, file), 'r') as f:
                bc = yaml.load(f, Loader=yaml.FullLoader)
                benchmark_constraints.append(bc)
        break

    for bc, core_buffer_size in itertools.product(benchmark_constraints, 2 ** np.arange(5, 12)):
        cluster_columns = [
            'core_num_mac',
            # 'core_noc_bw',
        ]
        fixed_columns = {
            'core_buffer_size': core_buffer_size,
            # 'core_buffer_bw': core_buffer_bw,  # fix this equals fixing num mac ...
        }
        try:
            for benchmark, layers in bc.items():
                print(cluster_columns, fixed_columns, benchmark)
                analyzer.plot_power_breakdown(cluster_columns, fixed_columns, benchmark=benchmark)
                break
        except:
            print(f"error: {cluster_columns} {fixed_columns}")
            traceback.print_exc()
            continue

    exit(1)

    for bc in benchmark_constraints:
        try:
            for benchmark, layers in bc.items():
                for wanted_layer in layers:
                    print("topi-1d", 'core_buffer_size', wanted_layer)
                    analyzer.plot_1d_topi('core_buffer_size', benchmark=benchmark, wanted_layers=[wanted_layer])
        except:
            traceback.print_exc()
            continue

    # for fixed buffer size, perf <- mac and noc?
    for bc, core_buffer_size in itertools.product(benchmark_constraints, 2 ** np.arange(5, 12)):
        cluster_columns = [
            'core_num_mac',
            'core_noc_bw',
        ]
        fixed_columns = {
            'core_buffer_size': core_buffer_size,
            # 'core_buffer_bw': core_buffer_bw,  # fix this equals fixing num mac ...
        }
        try:
            for benchmark, layers in bc.items():
                for wanted_layer in layers:
                    print(cluster_columns, fixed_columns, benchmark, wanted_layer)
                    analyzer.plot_cluster(cluster_columns, fixed_columns, benchmark=benchmark, wanted_layers=[wanted_layer])
        except:
            print(f"error: {cluster_columns} {fixed_columns}")
            traceback.print_exc()
            continue

    # for fixed noc, perf <- mac and buf?
    for bc, core_noc_bw  in itertools.product(benchmark_constraints, 2 ** np.arange(5, 12)):
        cluster_columns = [
            'core_num_mac',
            'core_buffer_size',
        ]
        fixed_columns = {
            'core_noc_bw': core_noc_bw,
            # 'core_buffer_bw': core_buffer_bw,  # fix this equals fixing num mac ...
        }
        try:
            for benchmark, layers in bc.items():
                for wanted_layer in layers:
                    print(cluster_columns, fixed_columns, benchmark, wanted_layer)
                    analyzer.plot_cluster(cluster_columns, fixed_columns, benchmark=benchmark, wanted_layers=[wanted_layer])
        except:
            print(f"error: {cluster_columns} {fixed_columns}")
            traceback.print_exc()
            continue