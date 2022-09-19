import os

from graph_parser import GraphParser
from outlog_parser import OutlogParser
from routing_parser import RoutingParser
from spec_parser import SpecParser

class TraceParser():
    """Collection of trace parsers on different logging files."""
    def __init__(self, graph_path, outlog_path, routing_path, spec_path) -> None:
        self.graph_parser = GraphParser(graph_path)
        self.outlog_parser = OutlogParser(outlog_path)
        self.spec_parser = SpecParser(spec_path)

        array_size = self.spec_parser.get_array_size()
        routing_func = self.spec_parser.get_routing_func()
        self.routing_parser = RoutingParser(routing_path, array_size, routing_func)