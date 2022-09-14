import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


from dataset.dataset import NoCDataset
from model.vanilla import VanillaModel

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

#------------------ Initalize Dataset ----------------------------#

dataset = NoCDataset()

num_examples = len(dataset)
num_train = int(num_examples * 0.9)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=2, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=2, drop_last=False)

#------------------ Initialize Model ----------------------------#

device = "cpu"
model = VanillaModel(h_dim=64, num_labels=4)