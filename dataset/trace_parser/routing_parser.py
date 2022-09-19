import os

class RoutingParser():
    """Parse Routing Information"""

    def __init__(self, routing_board_path, array_size, routing_func) -> None:
        self.routing_board_path = routing_board_path
        self.array_size = array_size
        self.routing_func = routing_func
        self.__multicast_routing = None


    def get_routing_hops(self, src, dst, pid):
        if self.__multicast_routing == None:
            # parse multicast on demand
            self.__parse_routing_board()
        if pid in self.__multicast_routing.keys():
            return self.__multicast_routing[pid]
        else:
            return self.__parse_unicast_hops(src, dst)


    def __parse_unicast_hops(self, src, dst):
        """Parse every hop from src PE to dst PE.
        src, dst: physical PE No.
        Returns: [one-hop edges]
        """
        edges = []
        k = self.array_size
        # order: (x, y)
        c2pt = lambda cid: (cid // k, cid % k)
        pt2c = lambda x, y: x * k + y

        x0, y0 = c2pt(src)
        x1, y1 = c2pt(dst)

        if self.routing_func == "src_routing":
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


    def __parse_routing_board(self):
        """Parse routing board for all the multicast packets' routing.
        Returns: {pid -> [one-hop edges]}
        """
        assert os.path.exists(self.routing_board_path)

        pid_to_edges = dict()
        with open(self.routing_board_path, "r") as f:
            line = f.readline()
            while len(line.strip("\n ")):
                parsed_line = [int(x) for x in line.split(" ")]
                pid = parsed_line[0]
                
                edges = []
                line = f.readline()
                while len(line.strip("\n ")):
                    u, v = [int(x) for x in line.split(" ")]
                    edges += self.__parse_unicast_hops(u, v, self.routing_func)
                    line = f.readline()

                pid_to_edges[pid] = edges

                f.readline()
                line = f.readline()

        self.__multicast_routing = pid_to_edges