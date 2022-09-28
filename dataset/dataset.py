from cProfile import label
from curses import raw
import os
import pickle as pkl
from dgl.data import DGLDataset
import numpy as np
import torch

import global_control as gc

class NoCDataset(DGLDataset):
    """#TODO: Could the dataset reside on memory?
    """
    def __init__(self, data_root=gc.data_root):
        super().__init__(name="NoC")
        assert os.path.exists(self.data_root)
        self.data_root = data_root

    def process(self):
        self.samples = [file for _, _, file in os.walk(self.data_root)][0]

    def __getitem__(self, i):
        """Returns: 
        graph: dgl.HeteroGraph
        congestion: Tensor(1, )
        """
        sample_path = os.path.join(self.data_root, self.samples[i])
        with open(sample_path, "rb") as f:
            graph, congestion = pkl.load(f)
        return graph, congestion

    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    dataset = NoCDataset()
    for i, g in enumerate(dataset):
        print(g)