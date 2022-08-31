import os
from pydoc import doc
import re
import networkx as nx
import yaml
from compiler.op_graph.micro_op_graph import MicroOpGraph

class TraceAnalyzer():
    """Analyze simresult to get some useful intermediate information
    """

    def __init__(self, taskname) -> None:
        self.taskname = taskname

        prj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        task_root = os.path.join(prj_root, "dataset/sim_result", taskname)

        self.op_graph_path = os.path.join(task_root, "op_graph.gpickle")
        self.out_log_path = os.path.join(task_root, "out.log")
        self.spatial_spec_path = os.path.join(task_root, "spatial_spec")
        self.routing_board_path = os.path.join(task_root, "routing_board")

        self.graph = nx.read_gpickle(self.op_graph_path)
        self.__initialize_graph()

        self.spec_info = None
        self.__initialize_spec_info()

        self.multicast_routing = None
        self.__initialize_multicast()


    def __get_packet_latency(self):
        """ (pid, dst) -> {src, start_cycle, end_cycle}
        """
        pid_to_latency = dict()

        # Handle src, start and end cycle
        with open(self.out_log_path, "r") as f:
            for line in f:
                parsed_line = self.__parse_instr(line)
                instr_type = parsed_line["type"]
                if instr_type == "NI.send":
                    pid = parsed_line["pid"]
                    for d in parsed_line["dst"]:
                        pid_to_latency[(pid, d)] = {
                            "src": parsed_line["core"],
                            "start_cycle": parsed_line["cycle"],
                            "end_cycle": -1  # to be filled
                        }
                elif instr_type == "NI.recv":
                    pid = parsed_line["pid"]
                    dst = parsed_line["core"]
                    pid_to_latency[(pid, dst)]["end_cycle"] = parsed_line["cycle"]
                elif instr_type == "CPU.sleep":
                    pass
                else:
                    pass
        
        return pid_to_latency

    
    def __initialize_graph(self):
        """Use spatial_sim to mark each flow's packets
        """

        # copied from toolchain's trace_generator
        raw_latency = self.__get_packet_latency()
        pkt_counter = 0

        op_graph = self.graph.get_graph()

        assert nx.is_directed_acyclic_graph(op_graph)
        for _, __, eattr in op_graph.edges(data=True):
            eattr["pkt"] = dict()

        node2pe = lambda x: op_graph.nodes[x]["p_pe"]

        for node in nx.topological_sort(op_graph):

            nattr = op_graph.nodes[node]
            iteraction_cnt = int(nattr["cnt"])

            # propagate data to data edges
            out_data_edges = [(u, v) for u, v, t in op_graph.out_edges(node, data="edge_type") if t == "data" and node2pe(u) != node2pe(v)]
            for _ in range(iteraction_cnt):
                flows = {op_graph.edges[e]["fid"] for e in out_data_edges}
                fid_to_pid = {fid: pid for fid, pid in zip(flows, range(pkt_counter, pkt_counter + len(flows)))}
                pkt_counter += len(flows)

                for u, v in out_data_edges:
                    fid = op_graph.edges[u, v]["fid"]
                    pid = fid_to_pid[fid]
                    op_graph.edges[u, v]["pkt"][pid] = raw_latency[(pid, node2pe(v))]

            # propagate control signals
            out_control_edges = [(u, v) for u, v, t in op_graph.out_edges(node, data="edge_type") if t == "control" and node2pe(u) != node2pe(v)]
            for u, v in out_control_edges:
                pid = pkt_counter
                pkt_counter += 1
                op_graph.edges[u, v]["pkt"][pid] = raw_latency[(pid, node2pe(v))]


    def get_src_pkts(self, src):
        """Return: {pid: [dsts]}
        """
        G = self.graph.get_graph()
        pid_to_dsts = dict()
        
        for v, eattr in G.out_edges(src, data=True):
            pkts = list(eattr["pkt"].keys())
            for pid in pkts:
                if pid not in pid_to_dsts.keys():
                    pid_to_dsts[pid] = []
                pid_to_dsts[pid].append(v)
        
        return pid_to_dsts


    def get_total_latency(self):
        line = None

        filesize = os.path.getsize(self.out_log_path)
        if filesize == 0:
            return None
        else:
            with open(self.out_log_path, 'rb') as fp: # to use seek from end, must uss mode 'rb'
                offset = -8            # initialize offset
                while -offset < filesize:  # offset cannot exceed file size
                    fp.seek(offset, 2)   # read # offset chars from eof(represent by number '2')
                    lines = fp.readlines()   # read from fp to eof
                    if len(lines) >= 2:  # if contains at least 2 lines
                        line = lines[-1]
                        break
                    else:
                        offset *= 2    # enlarge offset

        parsed_line = str(line).split('|')
        assert "Task Is Finished" in parsed_line[1]

        return int(parsed_line[0][2:].strip())


    def __initialize_spec_info(self):
        """
        Return: {infos}
        """
        self.spec_info = dict()

        with open(self.spatial_spec_path, "r") as f:
            for line in f:
                line = line.split("//")[0]
                line = line.strip('\n; ')
                if len(line) == 0:
                    continue
                parsed = line.split(" = ")
                assert len(parsed) == 2
                self.spec_info[parsed[0]] = parsed[1]


    def __initialize_multicast(self):
        """
        Assuming determinsitic routing.
        Return: {pid -> [one-hop edges]}

        You can tell if a packet is unicast once you get all multicast packets.
        """
        pid_to_edges = dict()
        routing_function = self.__get_routing_function()

        # multicast edges
        with open(self.routing_board_path, "r") as f:
            line = f.readline()
            while len(line.strip("\n ")):
                parsed_line = [int(x) for x in line.split(" ")]
                pid = parsed_line[0]
                
                edges = []
                line = f.readline()
                while len(line.strip("\n ")):
                    u, v = [int(x) for x in line.split(" ")]
                    edges += self.__parse_unicast_hops(u, v, routing_function)
                    line = f.readline()

                pid_to_edges[pid] = edges

                f.readline()
                line = f.readline()

        self.multicast_routing = pid_to_edges

    
    def get_routing_hops(self, src, dst, pid):
        if pid in self.multicast_routing.keys():
            return self.multicast_routing[pid]
        else:
            routing_func = self.__get_routing_function()
            return self.__parse_unicast_hops(src, dst, routing_func)


    def __parse_unicast_hops(self, src, dst, routing_function):
        """ (src, dst) -> [one-hop edges]
        """
        edges = []

        k = int(self.get_array_size())
        # order: (x, y)
        c2pt = lambda cid: (cid // k, cid % k)
        pt2c = lambda x, y: x * k + y

        x0, y0 = c2pt(src)
        x1, y1 = c2pt(dst)

        if routing_function == "src_routing":
            cur = src
            midpoints = []

            # first go along x
            if x1 >= x0:
                midpoints += [pt2c(x0 + i + 1, y0) for i in range(x1 - x0)]
            else:
                midpoints += [pt2c(x0 - i - 1, y0) for i in range(x0 - x1)]
            
            # then go along y
            if y1 >= y0:
                midpoints += [pt2c(x1, y0 + i + 1) for i in range(y1 - y0)]
            else:
                midpoints += [pt2c(x1, y0 - i - 1) for i in range(y0 - y1)]
            
            for c in midpoints:
                edges.append((cur, c))
                cur = c
        else:
            raise NotImplementedError

        return edges


    def __get_routing_function(self):
        return self.spec_info["routing_function"]
    
    
    def get_array_size(self):
        return int(self.spec_info["k"])

    
    def __parse_instr(self, line):
        """parse data line
        Return = {cycle, core, type, pid, (dst for send)}
        """
        ret = dict()
        parsed_line = line.split("|")
        if len(parsed_line) < 4:
            ret["type"] = "invalid"
        else:
            ret["cycle"] = int(parsed_line[0])
            ret["core"] = int(parsed_line[1].strip()[4:])
            parsed_instr = parsed_line[3].split()
            ret["type"] = parsed_instr[0].strip()
            ret["pid"] = int(parsed_instr[1])
            if ret["type"] == "NI.send":
                ret["dst"] = [int(d) for d in parsed_instr[2:]]
        return ret

    
    def get_num_layer(self):
        def get_layer(name):
            name = re.search("layer\d", name).group()
            name = re.search("\d", name).group()
            return int(name)
        
        G = self.graph.get_graph()
        layers = {get_layer(name) for _, name in G.nodes(data="layer")}
        return max(layers)

    
    def get_layers(self):
        G = self.graph.get_graph()
        return {name for _, name in G.nodes(data="layer")}