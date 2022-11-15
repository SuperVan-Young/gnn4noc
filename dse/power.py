import os
import yaml
import re
import numpy as np

import dse_global_control as gc

class PowerPredictor():
    """Predict a layer's power on a specific WSC architecture.
    """
    def __init__(self, **kwargs) -> None:
        self.task_root = kwargs['task_root']

    def run(self):
        """Give a detailed report of arch power estimation.
        """
        result = dict()
        result['mac'] = self.calc_mac_power()
        result['noc'] = self.calc_noc_power()
        result['sram'] = self.calc_sram_power()
        result['reticle'] = self.calc_reticle_power()
        return result


    def calc_mac_power(self):
        model_pattern = re.compile(r"^(.*)_layer\d+$")
        dims = ['C', 'M', 'N', 'P', 'Q', 'R', 'S']
        report = dict()

        benchmark_root = os.path.join(self.task_root, 'benchmark')
        for _, __, files in os.walk(benchmark_root):
            for file in files:
                with open(os.path.join(benchmark_root, file), 'r') as f:
                    bm = yaml.load(f, Loader=yaml.FullLoader)
                for model_full_name, layers in bm.items():
                    for layer_name, cores in layers.items():
                        model_name = model_pattern.match(layer_name).group(1)
                        layer_spec_path = os.path.join(gc.database_root, model_name, f"{layer_name}.yaml")
                        assert os.path.exists(layer_spec_path)
                        with open(layer_spec_path, 'r') as f:
                            layer_spec = yaml.load(f, Loader=yaml.FullLoader)
                        mac_total = np.prod([v for k, v in layer_spec['problem']['instance'] if k in dims])
                        report[layer_name] = mac_total
            break
        
        return report


    def calc_noc_power(self):
        pass

    def calc_sram_power(self):
        pass

    def calc_reticle_power(self):
        pass

