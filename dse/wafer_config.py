import os
import yaml
import json
import re
import numpy as np
from tqdm import tqdm
import time
from copy import deepcopy
import multiprocessing as mp
from itertools import product

from noc_spec import NoCSpec
import dse_global_control as gc
from sp_runner import run_focus, run_timeloop_mapper, run_timeloop_model

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.trace_parser.trace_parser import TraceParser
from dataset.predictor.lp_predictor import LinearProgrammingPredictor

class UnrollingConstraint():
    """Unroll 7-nested loop and generate valid timeloop constraint
    """
    #TODO: finish this part and start the experiment TONIGHT!
    def __init__(self, **kwargs) -> None:
        self.num_core = kwargs['num_core']
        self.core_num_mac = kwargs['core_num_mac']
        self.core_buffer_size = kwargs['core_buffer_size'] * 1024 // 16
        self.core_noc_bw = kwargs['core_noc_bw'] // 16
        self.dims = {
            "N": kwargs['N'],
            "M": kwargs['M'],
            "C": kwargs['C'],
            "P": kwargs['P'],
            "Q": kwargs['Q'],
            "R": kwargs['R'],
            "S": kwargs['S'],
        }
        self.unroll_records = {
            "dram_temporal": {k: 1 for k in self.dims.keys()},
            "dram_spatial": {k: 1 for k in self.dims.keys()},
            "glb_temporal": {k: 1 for k in self.dims.keys()},  # RS temp unroll_records here, but add constraint to weight buf
            "mac_spatial": {k: 1 for k in self.dims.keys()},
        }
        self.tensor_dims = {
            "weight_shape": ('M', "C", "R", "S"),
            "input_shape": ('N', "C", "P", "Q"),
            "output_shape": ('N', "M", "P", "Q"),
        }
        
        assert self.dims['N'] == 1


    def get_all_subfactors(self, num):
        """Get a number's all factors, return a set
        """
        res = set()
        for i in range(1, int(np.ceil(np.sqrt(float(num)))) + 1):
            if num % i == 0:
                res.add(i)
                res.add(num // i)
        return res


    def _get_remaining_dims(self, dim_name):
        d = np.prod([v[dim_name] for v in self.unroll_records.values()])
        return self.dims[dim_name] // int(d)


    def _get_utilized_global_buffer(self, core_unrolled_dims: dict):
        cur_tensor_sizes = [
            np.prod([
                core_unrolled_dims['M'],
                core_unrolled_dims['C'],
                core_unrolled_dims['R'],
                core_unrolled_dims['S'],
            ]), # weight
            np.prod([
                core_unrolled_dims['N'],
                core_unrolled_dims['C'],
                core_unrolled_dims['P'] + core_unrolled_dims['R'] - 1,
                core_unrolled_dims['Q'] + core_unrolled_dims['S'] - 1,
            ]), # input
            np.prod([
                core_unrolled_dims['N'],
                core_unrolled_dims['M'],
                core_unrolled_dims['P'],
                core_unrolled_dims['Q'],
            ]), # output
        ]
        return np.sum(cur_tensor_sizes)


    def _get_utilized_core(self, dram_unrolled_dims: dict):
        return np.prod(list(dram_unrolled_dims.values()))

    
    def add_to_unroll_records(self, unroll_type:str, unroll_factors:dict):
        assert unroll_type in self.unroll_records.keys()
        for k, v in unroll_factors.items():
            assert self.unroll_records[unroll_type][k] == 1
            self.unroll_records[unroll_type][k] = v


    def unroll_to_glb_temporal(self, dim_names:list, dims_alloc:list, verbose=False):
        """Unroll temporal dims to global buffer.
        Greadily find biggest unrolling factor product that buffer could hold.
        Doesn't guarantee to fully use dims_alloc.
        Returns: dict: dim_name -> unroll factor
        """
        # get unrolled dims within a core
        core_unrolled_dims = {k: 1 for k in self.dims.keys()}
        for unroll_type in ["glb_temporal", "mac_spatial"]:
            unroll_factors = self.unroll_records[unroll_type]
            for d in core_unrolled_dims.keys():
                core_unrolled_dims[d] *= unroll_factors[d]

        best_factors = tuple([1 for k in dim_names])

        # search for best factors
        subfactors = [self.get_all_subfactors(d) for d in dims_alloc]
        for factors in product(*subfactors):
            # get current tensor sizes
            cur_core_unrolled_dims = deepcopy(core_unrolled_dims)
            for dim_name, factor in zip(dim_names, factors):
                cur_core_unrolled_dims[dim_name] *= factor
            cur_tensor_sizes = self._get_utilized_global_buffer(cur_core_unrolled_dims)

            # check if tensor sizes could be held in global buffer
            if cur_tensor_sizes > self.core_buffer_size:
                continue

            # check if this is a better factor
            # factors at front of the list have higher priority and will be unrolled first
            if np.prod(factors) > np.prod(best_factors):
                best_factors = factors
            elif np.prod(factors) == np.prod(best_factors):
                is_better_factor = False
                for a, b in zip(factors, best_factors):
                    if a != b:
                        is_better_factor = (a > b)
                        break
                if is_better_factor: best_factors = factors
        
        if verbose:
            print(f"Unroll to glb temporal: ", [f"{dim_name}={factor}" for dim_name, factor in zip(dim_names, best_factors)])

        return {dim_name: factor for dim_name, factor in zip(dim_names, best_factors)}


    def unroll_to_dram_spatial(self, dim_names:list, dims_alloc:list, verbose=False):
        """Unroll to dram spatial.
        Greadily find biggest unrolling factor product that cores could hold.
        Returns: dict: dim_name -> unroll factor
        """
        best_factors = tuple([1 for k in dim_names])

        # search for best factors
        subfactors = [self.get_all_subfactors(d) for d in dims_alloc]
        for factors in product(*subfactors):
            # get current tensor sizes
            cur_dram_unrolled_dims = deepcopy(self.unroll_records['dram_spatial'])
            for dim_name, factor in zip(dim_names, factors):
                cur_dram_unrolled_dims[dim_name] *= factor
            if self._get_utilized_core(cur_dram_unrolled_dims) > self.num_core:
                continue

            # check if this is a better factor
            if np.prod(factors) >= np.prod(best_factors):
                best_factors = factors
        
        if verbose:
            print(f"Unroll to dram spatial: ", [f"{dim_name}={factor}" for dim_name, factor in zip(dim_names, best_factors)])

        return {dim_name: factor for dim_name, factor in zip(dim_names, best_factors)}


    def unroll_PQ(self):
        """Balance transmission and computation bottleneck.
        This method try different allocation to GLB, and fully utilize remaining cores,
        and determine the best PQ unrolling with both metric
        """
        
        dim_names = ['P', 'Q']
        assert self._get_remaining_dims('P') == self.dims['P']
        assert self._get_remaining_dims('Q') == self.dims['Q']

        best_glb_temporal_factors = tuple([1 for k in dim_names])
        best_dram_spatial_factors = tuple([1 for k in dim_names])
        best_latency = np.inf

        subfactors = [self.get_all_subfactors(self.dims['P']), self.get_all_subfactors(self.dims['Q'])]

        for glb_temporal_alloc_dims in product(*subfactors):
            glb_temporal_factors = self.unroll_to_glb_temporal(['P', 'Q'], glb_temporal_alloc_dims)
            dram_spatial_alloc_dims = [self.dims['P'] // glb_temporal_factors['P'], self.dims['Q'] // glb_temporal_factors['Q']]
            dram_spatial_factors = self.unroll_to_dram_spatial(['P', 'Q'], dram_spatial_alloc_dims)
            glb_temporal_factors.update({
                'M': self.unroll_records['glb_temporal']['M'],
                'C': self.unroll_records['glb_temporal']['C'],
                'R': self.unroll_records['glb_temporal']['R'],
                'S': self.unroll_records['glb_temporal']['S'],
            })
            dram_spatial_factors.update({
                'M': self.unroll_records['dram_spatial']['M'],
                'C': self.unroll_records['dram_spatial']['C'],
            })
            dram_temporal_factors = {
                'P': self.dims['P'] // (glb_temporal_factors['P'] * dram_spatial_factors['P']),
                'Q': self.dims['Q'] // (glb_temporal_factors['Q'] * dram_spatial_factors['Q']),
                'M': self.unroll_records['dram_temporal']['M'],
                'C': self.unroll_records['dram_temporal']['C'],
            }

            # calculate input, output, computation latency
            input_tile_per_core = np.prod([
                dram_temporal_factors['P'],
                dram_temporal_factors['Q'],
                dram_temporal_factors['M'],
                dram_temporal_factors['C'],
            ])
            input_num_core = np.prod([
                dram_spatial_factors['P'],
                dram_spatial_factors['Q'],
                dram_spatial_factors['C'],
            ])  # multicast!
            input_flit_size = np.prod([
                glb_temporal_factors['P'] + self.dims['R'] - 1,
                glb_temporal_factors['Q'] + self.dims['S'] - 1,
                glb_temporal_factors['C'],
            ])
            input_flit_size = input_flit_size // self.core_noc_bw
            input_flit_size = max(2, input_flit_size)
            input_latency = input_tile_per_core * input_num_core * input_flit_size

            output_tile_per_core = np.prod([
                dram_temporal_factors['P'],
                dram_temporal_factors['Q'],
                dram_temporal_factors['M'],
            ])
            output_num_core = np.prod([
                dram_spatial_factors['P'],
                dram_spatial_factors['Q'],
                dram_spatial_factors['C'],
                dram_spatial_factors['M'],
            ])
            output_flit_size = np.prod([
                glb_temporal_factors['P'],
                glb_temporal_factors['Q'],
                glb_temporal_factors['M'],
            ])
            output_flit_size = output_flit_size // self.core_noc_bw
            output_flit_size = max(2, output_flit_size)
            output_latency = output_tile_per_core * output_num_core * output_flit_size

            computation_tile_per_core = np.prod([
                dram_temporal_factors['P'],
                dram_temporal_factors['Q'],
                dram_temporal_factors['M'],
                dram_temporal_factors['C'],
            ])
            computation_delay = np.prod([
                glb_temporal_factors['P'],
                glb_temporal_factors['Q'],
                glb_temporal_factors['M'],
                glb_temporal_factors['C'],
                glb_temporal_factors['R'],
                glb_temporal_factors['S'],
            ])
            computation_latency = computation_tile_per_core * computation_delay

            total_latency = max([computation_latency, input_latency, output_latency])
            if total_latency < best_latency:
                total_latency = best_latency
                best_glb_temporal_factors = (glb_temporal_factors['P'], glb_temporal_factors['Q'])
                best_dram_spatial_factors = (dram_spatial_factors['P'], dram_spatial_factors['Q'])
        
        self.add_to_unroll_records('glb_temporal', {k: v for k, v in zip(dim_names, best_glb_temporal_factors)})
        self.add_to_unroll_records('dram_spatial', {k: v for k, v in zip(dim_names, best_dram_spatial_factors)})
        

    def unroll_to_mac_spatial(self, verbose=False):
        """Convert current dims in glb temporal to mac spatial
        """
        dim_names = ['C', 'M']
        best_factors = tuple([1 for k in dim_names])

        remaining_dims = [self.unroll_records['glb_temporal'][k] for k in dim_names]
        subfactors = [self.get_all_subfactors(v) for v in remaining_dims]

        for factors in product(*subfactors):
            num_utilized_mac = np.prod(factors)
            if num_utilized_mac > self.core_num_mac:
                continue
            if num_utilized_mac >= np.prod(best_factors) and num_utilized_mac % 2 == 0:
                best_factors = factors

        for dim_name, factor in zip(dim_names, best_factors):
            assert self.unroll_records['mac_spatial'][dim_name] == 1  # never unrolled before
            self.unroll_records['mac_spatial'][dim_name] = factor
            self.unroll_records['glb_temporal'][dim_name] //= factor

        if verbose:
            print(f"Unroll to mac spatial: ", [f"{dim_name}={factor}" for dim_name, factor in zip(dim_names, best_factors)])
            print(f"Shrinking glb temporal: ", [f"{dim_name}={self.unroll_records['glb_temporal'][dim_name]}" for dim_name in dim_names])

    def unroll_to_dram_temporal(self, dim_names):
        """At last, convert remaining dims to dram temporal.
        """
        for dim_name in dim_names:
            self.unroll_records['dram_temporal'][dim_name] = self._get_remaining_dims(dim_name)


    def dump_constraint(self, verbose=False):
        """Dump timeloop constraint. Only do this after all unrolling has been completed!
        """
        def get_factor_str(factors: dict):
            return " ".join(f"{l}={factor}" for l, factor in factors.items())

        for k in self.dims.keys():
            assert self._get_remaining_dims(k) == 1

        mapping_constraints = []

        # DRAM
        mapping_constraints.append({
            'target': 'DRAM',
            'type': 'temporal',
            'factors': get_factor_str(self.unroll_records['dram_temporal']),
            'permutation': 'RSPQNCM'
        })
        mapping_constraints.append({
            'target': 'DRAM',
            'type': 'spatial',
            'factors': get_factor_str(self.unroll_records['dram_spatial']),
        })

        # Global buffer
        mapping_constraints.append({
            'target': 'GlobalBuffer',
            'type': 'temporal',
            'factors': get_factor_str(self.unroll_records['glb_temporal']),
            'permutation': 'RSPQNCM',
        })

        pe_dims = ['C', 'M']
        is_double_pe = False
        for dim_name in pe_dims:
            if self.unroll_records['mac_spatial'][dim_name] % 2 == 0:
                is_double_pe = True
                glb_spatial = {k: 1 for k in pe_dims}
                glb_spatial[dim_name] = 2
                self.unroll_records['mac_spatial'][dim_name] //= 2
                mapping_constraints.append({
                    'target': 'GlobalBuffer',
                    'type': 'spatial',
                    'factors': get_factor_str(glb_spatial),
                })
                break
        assert is_double_pe, f"C={self.unroll_records['mac_spatial']['C']} M={self.unroll_records['mac_spatial']['M']}"

        # num mac
        mapping_constraints.append({
            'target': 'PEAccuBuffer',
            'type': 'spatial',
            'factors': get_factor_str({k: self.unroll_records['mac_spatial'][k] for k in pe_dims}),
        })

        # no temporal amplification in all subbuffers
        for sub_buffer in ['PEWeightRegs', 'PEInputBuffer', 'PEAccuBuffer', 'PEWeightBuffer']:
            mapping_constraints.append({
                'target': sub_buffer,
                'type': 'temporal',
                'factors': get_factor_str({k: 1 for k in self.dims.keys()})
            })

        if verbose:
            print(mapping_constraints)

        return mapping_constraints


    def run(self):
        """Unroll all dims in order and return the results.
        """
        verbose=False
        self.add_to_unroll_records("glb_temporal", self.unroll_to_glb_temporal(['R', 'S'], [self.dims['R'], self.dims['S']], verbose=verbose))  # weight stationary, all in!
        self.add_to_unroll_records("dram_spatial", self.unroll_to_dram_spatial(['M'], [self.dims['M']], verbose=verbose))  # best utlize cores for computation and multicasting
        self.add_to_unroll_records("glb_temporal", self.unroll_to_glb_temporal(['C', 'M'], [self.dims['C'], self._get_remaining_dims('M')], verbose=verbose))  # best utilize macs for computation, little help for transmission
        self.unroll_to_dram_temporal(['C', 'M'])
        self.unroll_to_mac_spatial(verbose=verbose)
        self.unroll_PQ()  # balance computation and transmission bottleneck
        self.unroll_to_dram_temporal(['P', 'Q'])

        return self.dump_constraint(verbose=verbose)


class WaferConfig():
    
    def __init__(self, **kwargs) -> None:
        # core config
        self.core_num_mac = kwargs['core_num_mac']
        self.core_buffer_bw = kwargs['core_buffer_bw']
        self.core_buffer_size = kwargs['core_buffer_size']

        # noc config
        self.core_noc_bw = kwargs['core_noc_bw']
        self.core_noc_vc = kwargs['core_noc_vc']
        self.core_noc_buffer_size = kwargs['core_noc_buffer_size']
        self.core_array_h = kwargs['core_array_h']
        self.core_array_w = kwargs['core_array_w']

        # reticle config
        self.reticle_bw = kwargs['reticle_bw']
        self.reticle_array_h = kwargs['reticle_array_h']
        self.reticle_array_w = kwargs['reticle_array_w']

        # memory config
        self.wafer_mem_bw = kwargs['wafer_mem_bw']

        assert self.core_num_mac >= 2, "core_num_mac < 2"

        self.task_root = os.path.join(gc.task_root, self._get_config_briefing())

    def run(self, dump_benchmark, invoke_timeloop_mapper, invoke_timeloop_model, invoke_focus, predict, verbose=False):
        """Run focus toolchain
        """
        task_root = os.path.join(gc.task_root, self._get_config_briefing())
        if not os.path.exists(task_root):
            os.mkdir(task_root)

        # dump all necessary configuration specs
        if dump_benchmark:
            self._dump_benchmark()
            self._dump_arch_config()
            self._dump_constraints_config()
            self._dump_modified_arch_config()

        if invoke_timeloop_mapper:
            if verbose: print(f"{self._get_config_briefing()}: Running timeloop mapper")

            for layers_root, dirs, files in os.walk(os.path.join(task_root, "layers")):
                layers = [os.path.join(layers_root, l) for l in dirs]
                with mp.Pool(processes=8) as pool:  # this is much faster
                    pool.map(run_timeloop_mapper, layers)
                break
            if verbose: print(f"{self._get_config_briefing()}: Finish timeloop mapper")
        
        if invoke_timeloop_model:
            if verbose: print(f"{self._get_config_briefing()}: Running timeloop model")
            for layers_root, dirs, files in os.walk(os.path.join(task_root, "layers")):
                layers = [os.path.join(layers_root, l) for l in dirs]
                with mp.Pool(processes=32) as pool:
                    pool.map(run_timeloop_model, layers)
                break
            if verbose: print(f"{self._get_config_briefing()}: Finish timeloop model")

        if predict:
            if verbose: print(f"{self._get_config_briefing()}: Predicting performance")
            self.predict_perf(invoke_focus)
            if verbose: print(f"{self._get_config_briefing()}: Finish predicting performance")

    def _get_config_briefing(self):
        briefs = [
            ('cm', self.core_num_mac),
            ('cbw', self.core_buffer_bw),
            ('csz', self.core_buffer_size),
            ('nvc', self.core_noc_vc),
            ('nbw', self.core_noc_bw),
            ('nsz', self.core_noc_buffer_size),
            ('nah', self.core_array_h),
            ('naw', self.core_array_w),
            ('rbw', self.reticle_bw),
            ('rah', self.reticle_array_h),
            ('raw', self.reticle_array_w),
            ('wmbw', self.wafer_mem_bw),
        ]
        return "_".join([f"{k}{v}" for k, v in briefs])

    def _dump_benchmark(self):
        """Dump benchmark config
        - Rename benchmark model, s.t. FOCUS will not overlap the result
        - Adjust core allocation, try to give as many cores as possible
        """
        benchmark_root = os.path.join(self.task_root, "benchmark")
        if not os.path.exists(benchmark_root):
            os.mkdir(benchmark_root)

        for benchmark_bu_root, dirs, files in os.walk(os.path.join(gc.dse_root, "benchmark")):
            for file in files:
                benchmark_bu_path = os.path.join(benchmark_bu_root, file)
                benchmark_constraint_path = os.path.join(os.path.join(gc.dse_root, "benchmark_constraint", file))
                assert os.path.exists(benchmark_constraint_path), f"Benchmark constraint {benchmark_constraint_path} should exist!"

                with open(benchmark_bu_path, 'r') as f:
                    benchmark_bu = yaml.load(f, Loader=yaml.FullLoader)
                assert len(benchmark_bu) == 1, "WaferConfig: only support single model performance prediction"
                for k, v in benchmark_bu.items():
                    benchmark_bu_name, benchmark_bu_layers = k, v

                get_layer_name = lambda x: list(x.keys())[0]
                get_layer_num_core = lambda x: list(x.values())[0]

                # core factor
                available_core_num = self.core_array_h * self.core_array_w * self.reticle_array_h * self.reticle_array_w
                demanding_core_num = np.sum([get_layer_num_core(l) for l in benchmark_bu_layers])
                core_factor = available_core_num / demanding_core_num
                core_factor = core_factor / gc.num_effective_model

                new_benchmark_name = f"{benchmark_bu_name}_{self._get_config_briefing()}"
                new_benchmark_config = [{get_layer_name(l): min(max(int(get_layer_num_core(l) * core_factor), 2), self.core_array_h * self.core_array_w)} for l in benchmark_bu_layers]

                # delete layers not in constraint
                with open(benchmark_constraint_path, "r") as f:
                    benchmark_constraint = yaml.load(f, Loader=yaml.FullLoader)
                assert len(benchmark_constraint) == 1, "WaferConfig: benchmark constraint multi model!"
                for k, v in benchmark_constraint.items():
                    benchmark_constraint = v
                    break
                new_benchmark_config = [l for l in new_benchmark_config if get_layer_name(l) in benchmark_constraint]

                new_benchmark = {new_benchmark_name : new_benchmark_config}

                benchmark_path = os.path.join(benchmark_root, file)
                with open(benchmark_path, "w") as f:
                    yaml.dump(new_benchmark, f)

    def _dump_arch_config(self):
        """Modify architecture configuration.
        """
        arch_path = os.path.join(gc.database_root, "arch_bu/cerebras_like.yaml")
        with open(arch_path, 'r') as f:
            arch_config = yaml.load(f, Loader=yaml.FullLoader)

        mac_datawidth = 16
        sram_total_depth = self.core_buffer_size * 1024 // mac_datawidth  # assume only one word in a row
        sram_total_nbanks = self.core_buffer_bw // mac_datawidth  # use nbanks to control bw

        # mac number
        # keep 2 PE for fanout, each PE should have 1/2 mac & reg
        pe_config = arch_config['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local']
        assert pe_config[-1]['name'] == "LMAC[0..3]"
        pe_config[-1]['name'] = f"LMAC[0..{(self.core_num_mac-1) // 2}]"
        pe_config[-1]['attributes']['datawidth'] = mac_datawidth
        assert pe_config[-2]['name'] == "PEWeightRegs[0..3]"
        pe_config[-2]['name'] = f"PEWeightRegs[0..{(self.core_num_mac-1) // 2}]"
        pe_config[-2]['attributes']['word-bits'] = mac_datawidth


        # buffer allocation:
        # keep GB for timeloop, assume size(GB) = size(WB) + size(IB)
        # - global_buffer: 1 size & bandwidth
        # - weight & input buffer : 1/2 size & bandwidth, 1/4 each
        # - accum buffer: 512B, 16 * mac bit/cycle, for free
        arch_config['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['name'] = 'PE[0..1]' # stay fanout ...
        ws_config = arch_config['architecture']['subtree'][0]['subtree'][0]['local']

        assert ws_config[0]['name'] == "GlobalBuffer"
        ws_config[0]['attributes']['word-bits'] = mac_datawidth
        ws_config[0]['attributes']['depth'] = sram_total_depth
        pe_config[2]['attributes']['nbanks'] = sram_total_nbanks
        pe_config[2]['attributes']['nports'] = 2  # read & write

        assert pe_config[0]['name'] == "PEInputBuffer"
        pe_config[0]['attributes']['word-bits'] = mac_datawidth
        pe_config[0]['attributes']['depth'] = sram_total_depth // 4
        pe_config[0]['attributes']['nbanks'] = sram_total_nbanks // 4

        assert pe_config[1]['name'] == "PEWeightBuffer"
        pe_config[1]['attributes']['word-bits'] = mac_datawidth
        pe_config[1]['attributes']['depth'] = sram_total_depth // 4
        pe_config[1]['attributes']['nbanks'] = sram_total_nbanks // 4

        assert pe_config[2]['name'] == "PEAccuBuffer"
        pe_config[2]['attributes']['word-bits'] = mac_datawidth
        pe_config[2]['attributes']['depth'] = 512 * 8 // mac_datawidth
        pe_config[2]['attributes']['nbanks'] = self.core_num_mac

        # dump arch config
        arch_root = os.path.join(self.task_root, "arch")
        if not os.path.exists(arch_root):
            os.mkdir(arch_root)

        arch_path = os.path.join(arch_root, "cerebras_like.yaml")
        with open(arch_path, 'w') as f:
            yaml.dump(arch_config, f)

    def _dump_constraints_config(self, verbose=False):
        """Add constraints for faster timeloop searching
        Cannot assure a valid mapping. FXXK TIMELOOP"""
        layers_root = os.path.join(self.task_root, "layers")
        if not os.path.exists(layers_root):
            os.mkdir(layers_root)

        for benchmark_root, dirs, files in os.walk(os.path.join(self.task_root, "benchmark")):
            for file in files:
                benchmark_path = os.path.join(benchmark_root, file)
                with open(benchmark_path, 'r') as f:
                    benchmark = yaml.load(f, Loader=yaml.FullLoader)
                for k, v in benchmark.items():
                    benchmark_name, benchmark_layers = k, v

                get_layer_name = lambda x: list(x.keys())[0]
                get_layer_num_core = lambda x: list(x.values())[0]

                for l in benchmark_layers:
                    num_core = get_layer_num_core(l)

                    parsed_layer_name = re.search(r"(^.*)_layer(\d+)$", get_layer_name(l))
                    model_name, layer_id = parsed_layer_name.group(1), parsed_layer_name.group(2)
                    layer_config_path = os.path.join(gc.focus_root, "database", model_name, f"{model_name}_layer{layer_id}.yaml")
                    with open(layer_config_path, 'r') as f:
                        layer_config = yaml.load(f, Loader=yaml.FullLoader)

                    unroller = UnrollingConstraint(
                        num_core=num_core,
                        core_num_mac=self.core_num_mac,
                        core_buffer_size=self.core_buffer_size,
                        core_noc_bw=self.core_noc_bw,
                        N=layer_config['problem']['instance']['N'],
                        C=layer_config['problem']['instance']['C'],
                        M=layer_config['problem']['instance']['M'],
                        P=layer_config['problem']['instance']['P'],
                        Q=layer_config['problem']['instance']['Q'],
                        R=layer_config['problem']['instance']['R'],
                        S=layer_config['problem']['instance']['S'],
                    )

                    constraint_bu_path = os.path.join(gc.focus_root, "database", "constraints", "simba_constraints_copy.yaml")
                    with open(constraint_bu_path, 'r') as f:
                        constraint = yaml.load(f, Loader=yaml.FullLoader)
                    constraint['mapspace_constraints']['targets'] = unroller.run()

                    layer_root = os.path.join(layers_root,  f"{get_layer_name(l)}_{get_layer_num_core(l)}")
                    if not os.path.exists(layer_root):
                        os.mkdir(layer_root)

                    constraint_path = os.path.join(layer_root, "constraints.yaml")
                    with open(constraint_path, 'w') as f:
                        yaml.dump(constraint, f)

    def _dump_modified_arch_config(self):
        """Copied from FOCUS toolchain.
        """
        with open(os.path.join(self.task_root, 'arch', 'cerebras_like.yaml')) as f:
            arch = yaml.load(f, Loader=yaml.FullLoader)

        for layers_root, dirs, files in os.walk(os.path.join(self.task_root, "layers")):
            for layer_dir in dirs:
                top_level_pe_cnt = int(re.search(r"^.*_(\d+)$", layer_dir).group(1))
                top_level_name = arch["architecture"]["subtree"][0]["subtree"][0]["name"]
                new_top_level_name = re.sub(r"0..\d+", "0.."+str(top_level_pe_cnt-1), top_level_name)
                arch["architecture"]["subtree"][0]["subtree"][0]["name"] = new_top_level_name

                new_top_arch_spec = os.path.join(layers_root, layer_dir, "modified_arch.yaml")
                with open(new_top_arch_spec, 'w') as f:
                    yaml.dump(arch, f)


    def predict_perf(self, invoke_focus):
        prediction_root = os.path.join(self.task_root, "prediction")
        if not os.path.exists(prediction_root):
            os.mkdir(prediction_root)

        for benchmark_root, __, files in os.walk(os.path.join(self.task_root, 'benchmark')):
            for file in files:
                benchmark_path = os.path.join(benchmark_root, file)

                with open(benchmark_path, 'r') as f:
                    benchmark = yaml.load(f, Loader=yaml.FullLoader)
                assert len(benchmark) == 1, "WaferConfig: only support single model performance prediction"
                for k, v in benchmark.items():
                    benchmark_name, benchmark_layers = k, v

                get_layer_name = lambda x: list(x.keys())[0]
                get_layer_num_core = lambda x: list(x.values())[0]

                # needless to copy
                # # copy timeloop mapper results to FOCUS dir``
                # layer_dirs = [f"{get_layer_name(l)}_{get_layer_num_core(l)}" for l in benchmark_layers]
                # for layer_dir in layer_dirs:
                #     src_dir = os.path.join(self.task_root, 'layers', layer_dir)
                #     dst_dir = os.path.join(gc.focus_root, 'buffer', 'timeloop-512g', layer_dir)
                #     if os.path.exists(dst_dir):
                #         os.system(f"rm -r {dst_dir}")
                #     os.system(f"cp -r {src_dir} {dst_dir}") 

                mode = "d"  # communication still use FOCUS'
                # core_array_size = max(self.core_array_h, self.core_array_w) * max(self.reticle_array_h, self.reticle_array_w)
                core_array_size = int(np.sqrt(np.sum([get_layer_num_core(l) for l in benchmark_layers]))) + 2  # eee, cannot run too big
                flit_size = self.core_noc_bw
                timeloop_buffer_path = os.path.join(self.task_root, "layers")
                if invoke_focus: run_focus(benchmark_path, core_array_size, flit_size, mode, timeloop_buffer_path, verbose=False, debug=False, timeout=3600)

                # trace parser
                taskname = f"{benchmark_name}_b1w{flit_size}_{core_array_size}x{core_array_size}"
                graph_path = gc.get_op_graph_path(taskname)
                routing_path = gc.get_routing_path(taskname)
                spec_path = gc.get_spec_path(taskname)
                assert os.path.exists(graph_path), f"graph_path {graph_path} doesn't exist!"
                assert os.path.exists(routing_path), f"routing_path {routing_path} doesn't exist!"
                assert os.path.exists(spec_path), f"spec_path {spec_path} doesn't exist!"
                trace_parser = TraceParser(
                    graph_path=graph_path,
                    outlog_path=None,
                    routing_path=routing_path,
                    spec_path=spec_path
                )

                noc_spec = NoCSpec(
                    trace_parser=trace_parser,
                    core_array_h=core_array_size,
                    core_array_w=core_array_size,
                    reticle_array_h=1,
                    reticle_array_w=1,
                    inter_reticle_bw=self.reticle_bw,
                    inter_core_bw=self.core_noc_bw,
                )  # useless for now

                predictor = LinearProgrammingPredictor(trace_parser, None)
                latencies = {
                    "prediction": {},
                    "theoretical": {},
                    "transmission": {},
                }
                for layer_name in trace_parser.graph_parser.get_layers():
                    latencies["prediction"][layer_name] = int(predictor.run(layer_name))
                    latencies["theoretical"][layer_name] = int(predictor.get_theoretical_latency(layer_name))
                    latencies['transmission'][layer_name] = predictor.get_data_transmission(layer_name)

                prediction_path = os.path.join(prediction_root, f"{benchmark_name}.json")
                with open(prediction_path, "w") as f:
                    f.write(json.dumps(latencies, indent=4))
            break

if __name__ == "__main__":
    wafer_config = WaferConfig(
        core_num_mac = 64,
        core_buffer_bw = 2048,
        core_buffer_size = 512,

        core_noc_bw = 4096,
        core_noc_vc = 4,
        core_noc_buffer_size = 4,
        core_array_h = 18,
        core_array_w = 19,

        reticle_bw = 1,
        reticle_array_h = 7, 
        reticle_array_w = 8,

        wafer_mem_bw = 4096, # testing!
    )
    wafer_config.run(dump_benchmark=True, invoke_timeloop_mapper=True, invoke_timeloop_model=True, invoke_focus=True, predict=True, verbose=True)