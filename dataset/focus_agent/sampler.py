import os
import random
import yaml
import re

def sample_within_range(distribution, min, max):
    """Sample from given distribution, but only keep valid value from (min, max)
    This prevents distribution skewing from brutal clamping.
    """
    x = distribution()
    cnt = 10
    while x < min or x > max:
        x = distribution()
        cnt -= 1
        if cnt == 0:
            return random.randint(min, max)  # return a default distribution
    return x

class LayerSample():
    """ An easy-to-use representation of a fake layer's computation info.
    """
    def __init__(self, args):
        self.params = None
        if isinstance(args, str):
            self.params = self.__parse_str(args)
        elif isinstance(args, dict):
            self.params = self.__parse_dict(args)
        else:
            raise NotImplementedError
        

    def __repr__(self):
        p = self.params
        s = f"cw{p['cnt_w']}_ci{p['cnt_i']}_co{p['cnt_o']}" +\
            f"_bw{p['broadcast_w']}_bi{p['broadcast_i']}" +\
            f"_fw{p['flit_w']}_fi{p['flit_i']}_fo{p['flit_o']}" +\
            f"_dw{p['delay_w']}_di{p['delay_i']}_do{p['delay_o']}" +\
            f"_n{p['worker']}"
        return s

    def dump(self, save_root, model_name=None):
        s = self.__repr__()
        if model_name == None:
            model_name = s
        savepath = os.path.join(save_root, f"{model_name}.yaml")
        data = {s: [{s: 2}]}
        with open(savepath, 'w') as f:
            yaml.dump(data, f)

    def __empty_params(self):
        empty_params = {
            'cnt_w': None,
            'cnt_i': None,
            'cnt_o': None,
            'flit_w': None,
            'flit_i': None,
            'flit_o': None,
            'delay_w': None,
            'delay_i': None,
            'delay_o': None,
            'broadcast_w': None,
            'broadcast_i': None,
            'worker': None,
        }
        return empty_params

    def __parse_str(self, s):
        params = self.__empty_params()
        short2full = {
            "cw": "cnt_w",
            "ci": "cnt_i",
            "co": "cnt_o",
            "bw": "broadcast_w",
            "bi": "broadcast_i",
            "fw": "flit_w",
            "fi": "flit_i",
            "fo": "flit_o",
            "dw": "delay_w",
            "di": "delay_i",
            "do": "delay_o",
            "n": "worker",
        }
        for t in s.split('_'):
            i = re.search('\d+', t).span()[0]
            key = short2full[t[:i]]
            val = int(t[i:])
            params[key] = val
        assert None not in params.values()
        return params

    def __parse_dict(self, args):
        params = self.__empty_params()
        for k in params.keys():
            params[k] = args[k]
        return params


class LayerSampler():
    def __init__(self) -> None:
        random.seed(114514)
        # set ranges, which should run successfully within 5 min
        self.ratio_range = 128
        self.cnt_range = 2 ** 12
        self.delay_range = 2 ** 13

    def get_random_sample(self):
        df_type, df_ratio = self._gen_dataflow()
        cnt = self._gen_cnt(df_type, df_ratio)
        delay = self._gen_delay(df_type, df_ratio)
        flit = self._gen_flit()
        worker = self._gen_worker()
        bcast = self._gen_broadcast()
        params = {
            'cnt_w': cnt[0],
            'cnt_i': cnt[1],
            'cnt_o': cnt[2],
            'flit_w': flit[0],
            'flit_i': flit[1],
            'flit_o': flit[2],
            'delay_w': delay[0],
            'delay_i': delay[1],
            'delay_o': delay[2],
            'broadcast_w': bcast[0],
            'broadcast_i': bcast[1],
            'worker': worker,
        }
        layer = LayerSample(params)
        return layer

    def _gen_dataflow(self):
        """Generate dataflow
        Return:
        - dataflow_type: str, uniform distribution
        - dataflow_ratio: int, gamma distribution
        """
        types = [
            "ns",
            "ws",
            "is",
            "os",
            "wis",
            "wos",
            "ios",
        ]
        dataflow_type = types[int(random.uniform(0, 7))]
        dataflow_ratio_sampler = lambda : int(random.gammavariate(alpha=2, beta=15))
        dataflow_ratio = sample_within_range(dataflow_ratio_sampler, 1, self.ratio_range)
        return dataflow_type, dataflow_ratio
    
    def _gen_delay(self, dataflow_type, dataflow_ratio):
        x = dataflow_ratio
        results = {
            "ns": (1, 1, 1),
            "ws": (x, 1, 1),
            "is": (1, x, 1),
            "os": (1, 1, x),
            "wis": (x, x, 1),
            "wos": (x, 1, x),
            "ios": (1, x, x),
        }
        distribution = lambda : int(2 ** random.uniform(0, 10))
        delay_factor = sample_within_range(distribution, 1, self.delay_range // x)
        ret = [i * delay_factor for i in results[dataflow_type]]
        return ret

    def _gen_cnt(self, dataflow_type, dataflow_ratio):
        x = dataflow_ratio
        results = {
            "ns": (1, 1, 1),
            "ws": (1, x, x),
            "is": (x, 1, x),
            "os": (x, x, 1),
            "wis": (1, 1, x),
            "wos": (1, x, 1),
            "ios": (x, 1, 1),
        }
        distribution = lambda : int(2 ** random.uniform(0, 10))
        cnt_factor = sample_within_range(distribution, 1, self.cnt_range // x)
        ret = [i * cnt_factor for i in results[dataflow_type]]
        return ret

    def _gen_broadcast(self):
        """Generate broadcast info.
        Use uniform distribution, two sources won't broadcast simultaneously.
        """
        types = ["n", "w", "i"]
        results = {
            "n": (0, 0),
            "w": (1, 0),
            "i": (0, 1),
        }
        bcast_type = types[int(random.uniform(0, 3))]
        return results[bcast_type]

    def _gen_flit(self):
        """Generate packet size.
        Use same gamma distribution for simplicity.
        """
        results = (
            int(random.gammavariate(alpha=2, beta=5)),
            int(random.gammavariate(alpha=2, beta=5)),
            int(random.gammavariate(alpha=2, beta=5)),
        )
        results = [max(i, 2) for i in results]
        return  results

    def _gen_worker(self):
        """Generate #workers
        Use uniform distribution between 1 and 16.
        """
        return int(random.uniform(1, 17))
        

if __name__ == "__main__":
    # Testing
    sampler = LayerSampler()
    layer = sampler.get_random_sample()
    print(sampler.get_random_sample())
    layer.dump(".")