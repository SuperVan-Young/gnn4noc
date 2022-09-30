from train import train
import multiprocessing as mp
from copy import deepcopy

if __name__ == "__main__":
    baseline_config = {
        "h_dim": 64,
        "n_hid": 2,
        "n_pred": 2,
        "pred_base" : 2.0,
        "pred_exp_min" : -2,
        "pred_exp_max" : 9,
    }

    larger_h_dim = deepcopy(baseline_config)
    larger_h_dim['h_dim'] = 128
    even_larger_h_dim = deepcopy(baseline_config)
    even_larger_h_dim['h_dim'] = 256

    less_pred_layer = deepcopy(baseline_config)
    less_pred_layer['pred_layer'] = 1

    smaller_pred_base = deepcopy(baseline_config)
    smaller_pred_base['pred_base'] = 2.0 ** 0.5
    smaller_pred_base['pred_exp_min'] *= 2
    smaller_pred_base['pred_exp_max'] *= 2

    configs = [
        baseline_config,
        # larger_h_dim,
        # even_larger_h_dim,
        # leaky_relu_activation,
        # elu_relu_activation,
        # gru_update,
        # set2set_pooling,
        # max_pooling,
        # avg_pooling,
        # less_pred_layer,
        # smaller_pred_base,
    ]

    with mp.Pool(processes=1) as pool:
        pool.map(train, configs, chunksize=1)