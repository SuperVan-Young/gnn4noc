import os

import yaml

dataset_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
benchmark_root = os.path.join(dataset_root, "benchmark")

if not os.path.exists(benchmark_root):
    os.mkdir(benchmark_root)

# task: available layers
tasks = {
    "alexnet": 8,
    "bert": 73,
    "bert-large": 145,
    "flappybird": 4,
    "inception": 95,
    "mnasnet": 53,
    "mobilenet_v3_large": 63,
    "mobilenet_v3_small": 54,
    "resnet50": 54,
    "resnext50_32x4d": 64,
    "ssd_r34": 51,
    "unet": 19,
    "vgg16": 16,
    "wide_resnet50_2": 64,
}

# treat different scale of network differently
cores = {
    "alexnet": [2, 4, 6, 8],
    "bert": [4, 6, 8, 10],
    "bert-large": [4, 6, 8, 10],
    "flappybird": [2, 4, 6, 8],
    "inception": [4, 6, 8, 10],
    "mnasnet": [4, 6, 8, 10],
    "mobilenet_v3_large": [4, 6, 8, 10],
    "mobilenet_v3_small": [4, 6, 8, 10],
    "resnet50": [4, 6, 8, 10],
    "resnext50_32x4d": [4, 6, 8, 10],
    "ssd_r34": [6, 8],
    "unet": [4, 6, 8],
    "vgg16": [8],
    "wide_resnet50_2": [4, 6, 8, 10],
}


for task, num_layers in tasks.items():
    for num_cores in cores[task]:
        layers = [{f"{task}_layer{i}": num_cores} for i in range(1, num_layers+1)]
        data = {f"{task}_{num_cores}": layers}

        benchmark_path = os.path.join(benchmark_root, f"{task}_{num_cores}.yaml")
        with open(benchmark_path, "w") as f:
            yaml.dump(data, f)
