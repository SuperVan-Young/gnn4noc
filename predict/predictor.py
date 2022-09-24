import os
import torch
import re
import sys
import random
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.trace_parser.trace_parser import TraceParser
from dataset.graph_generator.hyper import HyperGraphGenerator
from dataset.focus_agent.focus_agent import FocusAgent
from dataset.focus_agent.sampler import LayerSample, LayerSampler
import dataset.global_control as gc

class WorkerSampler(LayerSampler):
    def __init__(self) -> None:
        super().__init__()
        random.seed(time.time())

    def _gen_worker(self):
        """Generate #workers
        Use uniform distribution between 16 and 60.
        """
        return int(random.uniform(16, 60))

class LatencyPredictor():
    def __init__(self, model_path: str, fake_trace: bool, simulate: bool) -> None:
        self.model_path = model_path
        self.fake_trace = fake_trace
        self.simulate = simulate
        self.array_size = 8
        self.flit_size = 1024

    def predict_latency(self, benchmark_path: str):
        """Returns: predicted cycles, actual cycles (None if self.simulate == False)
        """
        print(f"Info: Running {benchmark_path}")

        # Run FOCUS to compile the benchmark
        focus_agent = FocusAgent(self.fake_trace, self.simulate)
        try:
            focus_agent.run_focus(benchmark_path, self.array_size, self.flit_size, timeout=300, verbose=True)
        except TimeoutError:
            print(f"Info: timeout when running {benchmark_path}, return invalid data")
            return None, None

        # Prepare trace parser
        taskname = os.path.split(benchmark_path)[1]
        taskname = re.search(r"(^.+).yaml", taskname).group(1)
        taskname = taskname + f"_b1w{self.flit_size}_{self.array_size}x{self.array_size}"

        graph_path = focus_agent.get_op_graph_path(taskname)
        spec_path = focus_agent.get_spec_path(taskname)
        routing_path = focus_agent.get_routing_path(taskname)
        outlog_path = focus_agent.get_outlog_path(taskname) if self.simulate else None
        trace_parser = TraceParser(graph_path, outlog_path, routing_path, spec_path)

        # prepare graph generator
        graph_generator = HyperGraphGenerator(trace_parser, predict=True)

        # load model
        with open(self.model_path, "rb") as f:
            model = torch.load(f)

        # calculate Model Latency
        predicted_latency = 0
        layers = trace_parser.graph_parser.get_layers()

        for layer in layers:
            graph = graph_generator.generate_graph(layer, batch=0)
            pred = model(graph).argmax()
            congestion = model.label_to_congestion(pred)

            G = trace_parser.graph_parser.get_graph(layer, 0)
            w = [u for u, attr in G.nodes(data=True) if attr['op_type'] == 'worker'][0]
            cnt, delay = G.nodes[w]['cnt'], G.nodes[w]['delay']
            latency = cnt * delay * (1 + congestion) + delay
            
            predicted_latency += int(latency)
        
        true_latency = trace_parser.outlog_parser.get_total_latency() if self.simulate else None

        return predicted_latency, true_latency

def test_more_workers():
    test_cnt = 300
    sampler = WorkerSampler()
    save_root = os.path.join(gc.gnn_root, "predict", "tmp")
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    model_path = os.path.join(gc.gnn_root, "train/log/vanilla_20220923095950/model.pth")
    predictor = LatencyPredictor(model_path, fake_trace=True, simulate=True)
    
    for i in range(test_cnt):
        layer = sampler.get_random_sample()
        layer.dump(save_root)
        layer_path = os.path.join(save_root, str(layer)+".yaml")

        predicted_latency, true_latency = predictor.predict_latency(layer_path)

        print(f"Layer config: {str(layer)}")
        print(f"Predicted latency: {predicted_latency}")
        print(f"true latency     : {true_latency}")
        print("")


if __name__ == "__main__":
    test_more_workers()