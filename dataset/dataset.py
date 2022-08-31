import os
import networkx as nx
import dgl
from dgl.data import DGLDataset

class NoCDataset(DGLDataset):
    """#TODO: Could the dataset reside on memory?
    """
    def __init__(self):
        super().__init__(name="NoC")
        

    def process(self):
        self.data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        assert os.path.exists(self.data_root)
        self.samples = (file for _, _, file in os.walk(self.data_root))


    def __getitem__(self, i):
        sample_path = os.path.join(self.data_root, self.samples[i])
        with open(sample_path, "r") as f:
            nx_graph = nx.read_gpickle(sample_path)
        g = dgl.from_networkx(nx_graph, 
            node_attrs=["delay", "in_latency", "out_latency", "op_type"], 
            edge_attrs=["size", "cnt", "route"])
        return g
    

    def __len__(self):
        return len(self.samples)
