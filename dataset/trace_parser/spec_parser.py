import os


class SpecParser():
    """Parse spec info."""

    def __init__(self, spec_path) -> None:
        self.spec_path = spec_path
        self.spec_info = None

    def __parse_spec(self):
        assert os.path.exists(self.spec_path)

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

    def get_routing_func(self):
        if self.spec_info == None:
            self.__parse_spec()
        return self.spec_info["routing_function"]
        
    def get_array_size(self):
        if self.spec_info == None:
            self.__parse_spec()
        return int(self.spec_info["k"])