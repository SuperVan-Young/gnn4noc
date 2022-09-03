from curses import raw
import os
import torch
import pickle as pkl
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
        self.samples = [file for _, _, file in os.walk(self.data_root)][0]


    def __getitem__(self, i):
        """Returns: 
        graph: dgl.HeteroGraph
        graph_info: Tensor(4,)
        congestion: Tensor(2,)
        """
        sample_path = os.path.join(self.data_root, self.samples[i])
        with open(sample_path, "r") as f:
            raw_data = pkl.load(sample_path, f)

        # build heterograph
        graph_data = {
            ('packet', 'pass', 'router'): self.__parse_edges(raw_data['packet_to_router']),
            ('router', 'connect', 'router'): self.__parse_edges(raw_data['router_to_router']),
        }
        g = dgl.heterograph(graph_data)

        # normalize cnt & delay
        cnt = raw_data['cnt']
        cnt_tensor = torch.tensor([cnt['wsrc'], cnt['insrc'], cnt['worker']]) / min([v for v in cnt.values()])
        delay = raw_data['delay']
        delay_tensor = torch.tensor(min([v for v in delay.values()]))
        graph_info = torch.concat(cnt_tensor, delay_tensor)

        # build congestion tensor
        congestion = torch.tensor([raw_data["w_congestion"], raw_data['in_congestion']])

        return g, graph_info, congestion
    

    def __len__(self):
        return len(self.samples)
        

    def __parse_edges(self, edges):
        """ Parse dict of edge lists.
        Return: source node tensor, dest node tensor
        """
        src, dst = [], []
        for s, l in edges.items():
            for d in l:
                src.append(s)
                dst.append(d)

        return torch.tensor(src), torch.tensor(dst)



if __name__ == "__main__":
    dataset = NoCDataset()
    for i, g in enumerate(dataset):
        print(g)