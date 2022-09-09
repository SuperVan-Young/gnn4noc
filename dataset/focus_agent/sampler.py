import os
import random

class DataSampler():
    def __init__(self) -> None:
        random.seed(114514)

    def get_random_sample(self):
        df_type, df_ratio = self._gen_dataflow()
        cnt = self._gen_cnt(df_type, df_ratio)
        delay = self._gen_delay(df_type, df_ratio)
        flit = self._gen_flit()
        worker = self._gen_worker()
        bcast = self._gen_broadcast()
        ret = {
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
        layer = self._encode_string(ret)
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
        dataflow_ratio = int(random.gammavariate(alpha=2, beta=15))
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
        return results[dataflow_type]

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
        return results[dataflow_type]

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
        Use same gamma distribution for simplicity
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

    def _encode_string(self, args):
        """Encode args into string"""
        s = f"cw{args['cnt_w']}_ci{args['cnt_i']}_co{args['cnt_o']}" +\
            f"_bw{args['broadcast_w']}_bi{args['broadcast_i']}" +\
            f"_fw{args['flit_w']}_fi{args['flit_i']}_fo{args['flit_o']}" +\
            f"_dw{args['delay_w']}_di{args['delay_i']}_do{args['delay_o']}" +\
            f"_n{args['worker']}"
        return s

if __name__ == "__main__":
    # Testing
    sampler = DataSampler()
    print(sampler.get_random_sample())