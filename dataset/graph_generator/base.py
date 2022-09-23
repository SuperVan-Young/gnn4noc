import os
import sys
sys.path.append("..")

import pickle as pkl
import torch
from collections import UserDict
import networkx as nx

from trace_parser.trace_parser import TraceParser

class SmartDict(UserDict):
    """Automatically record new key's ID incrementally from 0
    """
    def __init__(self):
        super().__init__()
        self.__cnt = 0

    def __getitem__(self, x):
        if x not in self.data.keys():
            self.data[x] = self.__cnt
            self.__cnt += 1
        return self.data[x]

    def __len__(self):
        return self.__cnt


def binarize_float(tensor, bit):
    """Binarize float into integer, represented in vector
    Input: tensor(*)
    Return: tensor(*, bit)
    """
    quantity_tensors = []

    for i in range(bit):
        q = tensor % (2 ** (i+1))
        q = torch.div(q, 2 ** i, rounding_mode='floor')
        quantity_tensors.append(q)

    bin_tensor = torch.stack(quantity_tensors, dim=-1).float()
    return bin_tensor


class GraphGenerator():
    """Generating graph from parser.
    This is simply a base class.
    """

    def __init__(self, parser: TraceParser, predict: bool) -> None:
        self.parser = parser
        self.predict = predict

    def generate_graph(self, layer=None, batch=None):
        """Generate subgraph if layer and batch are given, else the whole graph.
        """
        G = self.__load_graph()
        graph = self._gen_graph(G)
        return graph

    def generate_label(self, layer=None, batch=None):
        assert self.predict == False, "Graph generator: no access to label during prediction."
        G = self.__load_graph(layer, batch)
        label = self._gen_label(G)
        return label

    def save_data(self, save_path, graph, label=None):
        saving = graph if self.predict else (graph, label)
        with open(save_path, "wb") as f:
            pkl.dump(saving, f)

    def __load_graph(self, layer=None, batch=None):
        if layer != None and batch != None:
            G = self.parser.graph_parser.get_graph(layer, batch)
        else:
            if layer != None:
                raise Warning(f"Graph Generator: layer {layer} without specific batch, \
                    use whole graph instead.")
            if batch != None:
                raise Warning(f"Graph Generator: batch {batch} without specific layer, \
                    use whole graph instead.")
            G = self.parser.graph_parser.get_graph(None, None)
        return G

    def _gen_graph(self, G:nx.DiGraph):
        raise NotImplementedError("Base Graph Generator doesn't implement _gen_graph")

    def _gen_label(self, G:nx.DiGraph):
        raise NotImplementedError("Base Graph Generator doesn't implement _gen_label")