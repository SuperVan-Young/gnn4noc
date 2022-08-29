import os
import networkx as nx
import yaml
from micro_op_graph import MicroOpGraph

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
        

    def get_op_graph(self):
        graph = nx.read_gpickle(self.op_graph_path)
        return graph


    def get_packet_latency(self):
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


    def get_routing_path(self):
        """
        Assuming determinsitic routing.
        Return: pid -> [one-hop edges]
        """
        pass

    def __parse_multicast_path(self):
        """
        """
    

    def get_spec_info(self):
        """
        Return: {infos}
        """
        

        pass

    
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