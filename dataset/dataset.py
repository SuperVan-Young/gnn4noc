from curses import raw
import os
import pickle as pkl
from dgl.data import DGLDataset

class NoCDataset(DGLDataset):
    """#TODO: Could the dataset reside on memory?
    """
    def __init__(self):
        super().__init__(name="NoC")
        

    def process(self):
        self.data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        assert os.path.exists(self.data_root)
        self.samples = [file for _, _, file in os.walk(self.data_root)][0]


    def __getitem__(self, i):
        """Returns: 
        graph: dgl.HeteroGraph
        congestion: Tensor(2,)
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