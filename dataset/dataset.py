from cProfile import label
from curses import raw
import os
import pickle as pkl
from dgl.data import DGLDataset
import numpy as np
import torch

class NoCDataset(DGLDataset):
    """#TODO: Could the dataset reside on memory?
    """
    def __init__(self, label_min, label_max):
        super().__init__(name="NoC")
        self.num_classes = label_max - label_min
        self.__ref_labels = torch.tensor(2.0 ** np.arange(label_min, label_max-1))

    def process(self):
        self.data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        assert os.path.exists(self.data_root)
        self.samples = [file for _, _, file in os.walk(self.data_root)][0]


    def __getitem__(self, i):
        """Returns: 
        graph: dgl.HeteroGraph
        congestion: Tensor(num_classes,)
        """
        sample_path = os.path.join(self.data_root, self.samples[i])
        with open(sample_path, "rb") as f:
            graph, congestion = pkl.load(f)
        label = self.__c2l(congestion)

        return graph, label
    

    def __len__(self):
        return len(self.samples)


    def __c2l(self, congestion):
        """Convert Congestion into #num_labels category"""
        label = (congestion < self.__ref_labels).int()
        label = torch.cat([label, torch.ones(1,)])
        label = torch.argmax(label)
        return label


if __name__ == "__main__":
    dataset = NoCDataset()
    for i, g in enumerate(dataset):
        print(g)