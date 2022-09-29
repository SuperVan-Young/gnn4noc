import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataset.dataset import NoCDataset
from model.hyper_graph_model import HyperGraphModel
from logger import Logger
from tqdm import tqdm

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

example_model_config = {
    "h_dim": 64,
    "activation":'ReLU',
    "num_mp": 2,
    "update": "activate",
    "readout": "sum",
    "pred_layer": 2,
    "pred_base" : 2.0,
    "pred_exp_min" : -2,
    "pred_exp_max" : 9,
}

def train(model_config):
    #------------------ Global Config --------------------------------#

    device = "cpu"
    verbosity = 1  # 0 for debugging

    # training configs
    epoches = 70
    learning_rate = 3e-4
    batch_size = 4

    #------------------ Initalize Dataset ----------------------------#

    data_root = "/home/xuechenhao/gnn4noc/dataset/data/router/"
    dataset = NoCDataset(data_root=data_root)

    num_examples = len(dataset)
    num_train = int(num_examples * 0.9)

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False)

    #------------------ Initialize Model ----------------------------#

    model = HyperGraphModel(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #------------------ Initialize Logger ----------------------------#

    logger = Logger("Hyper Graph Model", model, verbosity=verbosity)
    logger.info(f"data root = {data_root}")
    for key, val in model_config.items():
        logger.info(f"{key} = {val}")

    #------------------ Training Model ----------------------------#

    def feed_data(dataloader, train=False):
        losses = []
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="train" if train else "test")

        for batched_graph, congestion in pbar:
            pred = model(batched_graph)
            labels = model.congestion_to_label(congestion)
            loss = F.cross_entropy(pred, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            correct += (pred.argmax(1) == labels).sum().item()
            total += len(labels)

            pbar.postfix = f"acc = {correct / total:.2%}"

        avg_loss = np.average(losses)
        avg_acc = correct / total
        return avg_loss, avg_acc

    for e in range(epoches):
        logger.info(f"Epoch = {e}")

        # train
        train_loss, train_acc = feed_data(train_dataloader, train=True)
        logger.info(f"Training loss = {train_loss:.2f}")
        logger.info(f"Training acc  = {train_acc:.2%}")
        logger.append_train_loss(train_loss)
        logger.append_train_acc(train_acc)

        # test loss and accuracy
        test_loss, test_acc = feed_data(test_dataloader, train=False)
        logger.info(f"Testing loss = {test_loss:.2f}")
        logger.info(f"Testing acc  = {test_acc:.2%}")
        logger.append_test_loss(test_loss)
        logger.append_test_acc(test_acc)

    # --------------------- Dump Result -----------------------------# 

    logger.dump_model()
    logger.plot_curve("train_loss")
    logger.plot_curve("test_loss")
    logger.plot_curve("train_acc")
    logger.plot_curve("test_acc")

    logger.info("Finished training model.")

    #-------------------- Test congestion accuracy ----------------------------#

    logger.info("Testing congestion prediction performance.")

    dataset.use_label = False
    congestion_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=1, drop_last=False)

    for g, cong in congestion_dataloader:
        pred = model(g)
        pred = model.label_to_congestion(pred.argmax(1).item())
        logger.info(f"Ground Truth = {cong.item()}, Pred = {pred}")

if __name__ == "__main__":
    train(example_model_config)