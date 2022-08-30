# GNN4NoC

## Prerequisites
- FOCUS scheduler
- DGL

## Build Dataset
Run the following command:
```
cd dataset/scripts
python dump_benchmark.py
bash run_focus.sh
bash fetch_simresult.sh
python convert_data.py
```