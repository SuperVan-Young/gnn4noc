import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_space import parse_design_point

# power constants
MAC_DYNAMIC_ENERGY = 35.3 + 16.679       # pJ
MAC_STATIC_POWER = (0.12 + 0.053) * 1e3  # W

NOC_CHANNEL_FACTOR = 0.15                # pJ / bit / mm

CEREBRAS_RETICLE_CHANNEL_ENERGY = 0.25   # pJ / bit
DOJO_RETICLE_CHANNEL_ENERGY = 1.25       # pJ / bit

def build_power_table(sram_table_path, noc_table_path, output_path):
    with open(sram_table_path, 'r') as f:
        sram_table = json.load(f)
    with open(noc_table_path, 'r') as f:
        noc_table = json.load(f)

    output_table = dict()

    for dp in sram_table.keys():
        table = dict()

        parsed_dp_list = dp.strip("[]").split(", ")
        parsed_dp_list = [int(float(v)) for v in parsed_dp_list]
        parsed_dp = parse_design_point(parsed_dp_list)

        table['mac_dynamic_energy'] = MAC_DYNAMIC_ENERGY
        table['mac_static_power'] = MAC_STATIC_POWER

        table['sram_static_power'] = sram_table[dp]['static_power']
        table['sram_read_energy'] = sram_table[dp]['read_power']
        table['sram_write_energy'] = sram_table[dp]['write_power']

        core_array_size = max(parsed_dp['core_array_h'], parsed_dp['core_array_w'])
        noc_length = 300 / core_array_size  # mm

        table['noc_channel_energy'] = noc_length * NOC_CHANNEL_FACTOR
        table['noc_static_power'] = noc_table[dp]['static_power']

        table['reticle_channel_energy'] = CEREBRAS_RETICLE_CHANNEL_ENERGY if 'cerebras' in output_path else DOJO_RETICLE_CHANNEL_ENERGY

        table['noc_bw'] = parsed_dp['core_noc_bw']
        table['reticle_bw'] = parsed_dp['reticle_bw']

        output_table[str(parsed_dp_list)] = table

    with open(output_path, "w") as f:
        f.write(json.dumps(output_table, indent=4))

if __name__ == "__main__":
    build_power_table('power_table.json', 'noc_power.json', 'cerebras.json')
    build_power_table('power_table_Dojo.json', 'noc_power_Dojo.json', 'dojo.json')