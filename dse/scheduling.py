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

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class UnrollingConstraint():
    """Unroll 7-nested loop and generate valid timeloop constraint
    """
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
            # make sure C and M gets even
            if ('C' in dim_names or 'M' in dim_names) and np.prod(factors) % 2 != 0:
                continue

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