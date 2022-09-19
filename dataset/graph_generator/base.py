import os
import sys
import pickle as pkl
sys.path.append("..")

from trace_parser.trace_parser import TraceParser


class GraphGenerator():
    """Generating graph from parser."""

    def __init__(self, parser: TraceParser, predict: bool) -> None:
        self.parser = parser
        self.predict = predict

    def generate_graph(self, layer=None, batch=None):
        """Generate subgraph if layer and batch are given, else the whole graph.
        """
        graph = self.__gen_graph(layer, batch)
        return graph

    def save_data(self, save_path, graph, label=None):
        saving = graph if self.predict else (graph, label)
        with open(save_path, "wb") as f:
            pkl.dump(saving, f)

    def __gen_graph(self, layer=None, batch=None):
        