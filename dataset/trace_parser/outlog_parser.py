import os

class OutlogParser():
    def __init__(self, out_log_path) -> None:
        self.out_log_path = out_log_path
        self.__pid_to_latency = None

    def get_latency(self, src, dst, pid):
        if self.__pid_to_latency == None:
            self.__parse_out_log()
        return self.__pid_to_latency[(pid, dst)]

    def get_total_latency(self):
        assert os.path.exists(self.out_log_path)

        line = None
        filesize = os.path.getsize(self.out_log_path)
        if filesize == 0:
            raise RuntimeError(f"Empty out log {self.out_log_path}")
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
        total_latency = int(parsed_line[0][2:].strip())

        return total_latency

    def __parse_out_log(self):
        """ (pid, dst) -> {start_cycle, end_cycle}
        - dst: p_pe. one packet could be multicast to different dsts.
        """
        assert os.path.exists(self.out_log_path)
        pid_to_latency = dict()
        with open(self.out_log_path, "r") as f:
            for line in f:
                parsed_line = self.__parse_instr(line)
                instr_type = parsed_line["type"]
                if instr_type == "NI.send":
                    pid = parsed_line["pid"]
                    for d in parsed_line["dst"]:
                        pid_to_latency[(pid, d)] = {
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
        
        self.__pid_to_latency = pid_to_latency

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