from train import train
import multiprocessing as mp
from copy import deepcopy

if __name__ == "__main__":
    baseline_config = {
        "h_dim": 64,
        "n_hid": 2,
        "n_pred": 2,
        "message_passing": "vanilla",
        "pred_base" : 2.0,
        "pred_exp_min" : -1,
        "pred_exp_max" : 9,
    }

    more_hid_layer = deepcopy(baseline_config)
    more_hid_layer['hid_layer'] = 3

    hgt = deepcopy(baseline_config)
    hgt['message_passing'] = 'HGT'

    less_pred_layer = deepcopy(baseline_config)
    less_pred_layer['pred_layer'] = 1

    smaller_pred_base = deepcopy(baseline_config)
    smaller_pred_base['pred_base'] = 2.0 ** 0.5
    smaller_pred_base['pred_exp_min'] *= 2
    smaller_pred_base['pred_exp_max'] *= 2

    configs = [
        baseline_config,
        # more_hid_layer,
        hgt,
    ]

    with mp.Pool(processes=1) as pool:
        pool.map(train, configs, chunksize=1)