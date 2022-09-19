import os
import torch
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.trace_parser.trace_parser import TraceParser
from dataset.graph_generator.base import GraphGenerator
from dataset.focus_agent.focus_agent import FocusAgent
import dataset.global_control as gc

class LatencyPredictor():
    def __init__(self, benchmark_path, model_path, fake_trace) -> None:
        self.benchmark_path = benchmark_path
        self.model_path = model_path
        self.fake_trace = fake_trace
        self.array_size = 8
        self.flit_size = 1024

    def get_taskname(self):
        taskname = os.path.split(self.benchmark_path)[1]
        taskname = re.search(r"(^.+).yaml", taskname).group(1)
        taskname = f"{taskname}_b1w{self.flit_size}_{self.array_size}x{self.array_size}"
        return taskname

    def predict_latency(self):
        """Returns: predicted cycles.
        """
        # Run FOCUS to compile the benchmark
        focus_agent = FocusAgent(self.fake_trace, simulate=False)
        focus_agent.run_focus(self.benchmark_path, self.array_size, self.flit_size, timeout=300, verbose=True)

        # Prepare trace parser
        taskname = self.get_taskname()
        graph_path = focus_agent.get_op_graph_path(taskname)
        spec_path = focus_agent.get_spec_path(taskname)
        routing_path = focus_agent.get_routing_path(taskname)
        outlog_path = None  # prediction
        trace_parser = TraceParser(graph_path, outlog_path, routing_path, spec_path)

        graph_generator = GraphGenerator(trace_parser, predict=True)
        with open(self.model_path, "rb") as f:
            model = torch.load(f)

        layers = trace_parser.graph_parser.get_layers()
        total_latency = 0

        for layer in layers:
            graph = graph_generator.generate_graph(layer, batch=0)
            pred = model(graph).argmax()
            congestion = model.label_to_congestion(pred)

            # calculate predicted latency
            G = trace_parser.graph_parser.get_graph(layer, 0)
            w = [u for u, attr in G.nodes(data=True) if attr['op_type'] == 'worker'][0]
            cnt, delay = G.nodes[w]['cnt'], G.nodes[w]['delay']
            latency = cnt * delay * (1 + congestion) + delay
            
            total_latency += latency

        return latency

if __name__ == "__main__":
    benchmark_path = os.path.join(gc.gnn_root, "predict", "cw1_ci1_co1_bw0_bi0_fw2_fi2_fo4_dw1_di1_do1_n1.yaml")
    model_path = os.path.join(gc.gnn_root, "train", "log", "vanilla_1663153664", "model.pth")

    latency_predictor = LatencyPredictor(benchmark_path, model_path, True)
    latency = latency_predictor.predict_latency()
    print(f"Predicted latency = {latency}")