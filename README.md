# GNN4NoC

## Prerequisites
- python >= 3.8
- FOCUS scheduler (if building dataset is needed)
- DGL
- pytorch

## Build Dataset
Run the following command:
```
cd dataset/scripts
python dump_benchmark.py
bash run_focus.sh
bash fetch_simresult.sh
python convert_data.py
```