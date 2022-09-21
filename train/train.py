import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataset.dataset import NoCDataset
from model.vanilla import VanillaModel
from logger import Logger

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

#------------------ Global Config --------------------------------#

device = "cpu"
verbosity = 1  # 0 for debugging

# training configs
epoches = 50
learning_rate = 3e-4
batch_size = 4

# model configs
h_dim = 64
base = 2.0
label_min = -2
label_max = 9

#------------------ Initalize Dataset ----------------------------#

dataset = NoCDataset()

num_examples = len(dataset)
num_train = int(num_examples * 0.9)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False)

#------------------ Initialize Model ----------------------------#

model = VanillaModel(h_dim=64, label_min=label_min, label_max=label_max).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#------------------ Initialize Logger ----------------------------#

logger = Logger("vanilla", model, verbosity=verbosity)

logger.info(f"hidden dim = {h_dim}")
logger.info(f"base = {base}")
logger.info(f"label_min = {label_min}")
logger.info(f"label_max = {label_max}")
logger.info(f"epoches = {epoches}")
logger.info(f"learning rate = {learning_rate}")

#------------------ Training Model ----------------------------#

def feed_data(dataloader, train=False):
    losses = []
    correct = 0
    total = 0

    for batched_graph, congestion in dataloader:
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
        
        logger.debug(f"dataloader: {total} / {len(dataloader) * batch_size}")

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