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

label_min = -2
label_max = 8
num_labels = label_max - label_min  # [2 ** min, 2 ** max)

device = "cpu"
epoches = 30
batch_size = 4
verbosity = 0  # 0 for debugging

#------------------ Initalize Dataset ----------------------------#

dataset = NoCDataset(label_min, label_max)

num_examples = len(dataset)
num_train = int(num_examples * 0.9)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False)

#------------------ Initialize Model ----------------------------#

model = VanillaModel(h_dim=64, num_labels=num_labels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
logger = Logger("vanilla", model, verbosity=verbosity)

def feed_data(dataloader, train=False):
    losses = []
    correct = 0
    total = 0

    for batched_graph, labels in dataloader:
        pred = model(batched_graph)
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
    
#------------------ Training Model ----------------------------#

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