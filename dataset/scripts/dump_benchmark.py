import os

import yaml

dataset_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
benchmark_root = os.path.join(dataset_root, "benchmark")

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

cores = [2, 4, 6, 8]

for num_cores in cores:
    for task, num_layers in tasks.items():
        layers = [{f"{task}_layer{i}": num_cores} for i in range(1, num_layers+1)]
        data = {task: layers}

        benchmark_path = os.path.join(benchmark_root, f"{task}_{num_cores}.yaml")
        with open(benchmark_path, "w") as f:
            yaml.dump(data, f)

# debug
with open(os.path.join(benchmark_root,"alexnet_2.yaml"), "r") as f:
    data = yaml.load(f)
    print(data)
