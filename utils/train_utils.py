from pathlib import Path
import torch
from typing import Dict, List
import math
import os
import matplotlib.pyplot as plt
import torch
import json
import os
import random
import numpy as np
import subprocess
import re
import transformers
import time
from sklearn.metrics import classification_report, confusion_matrix

def set_seed(seed: int):
    """set down all random factors for reproducing results in future"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_time():
    return time.strftime('%b%d_%H%M_%S', time.localtime())

def get_optimizer(model: torch.nn.Module, args=None):
    if args is None:
        return torch.optim.AdamW(
            model.parameters(), 
            lr=0.001, betas=(0.9, 0.999), eps=1e-08, 
            weight_decay=0.01, amsgrad=False
        )
    else:
        raise NotImplementedError

    return 

def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


logger_base = Path(__file__).resolve().parent.parent / 'log'



def plot_loss(train_log: List[Dict], keys: List[str] = ["loss"], train_id = f'None') -> None:
        
    for key in keys:
        steps, metrics = [], []
        for step_info in train_log:
            if key in step_info:
                steps.append(step_info["step"])
                metrics.append(step_info[key])

        plt.figure()
        plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed")
        plt.title("training {} of {}".format(key, train_id))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        figure_path = os.path.join(logger_base / train_id, "training_{}.png".format(key.replace("/", "_")))
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)

def select_gpu():
    try:
        nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE).stdout.decode()
    except UnicodeDecodeError:
        nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE).stdout.decode("gbk")
    used_list = re.compile(r"(\d+)MiB\s+/\s+\d+MiB").findall(nvidia_info)
    used = [(idx, int(num)) for idx, num in enumerate(used_list)]
    sorted_used = sorted(used, key=lambda x: x[1])
    print(f'auto select gpu-{sorted_used[0][0]}, sorted_used: {sorted_used}')
    return sorted_used[0][0]

def set_device(gpu) -> str:
    assert gpu < torch.cuda.device_count(), f'gpu {gpu} is not available'
    if not torch.cuda.is_available():
        return 'cpu'
    if gpu == -1:  gpu = select_gpu()
    return f'cuda:{gpu}'

def eval_classify(all_labels, all_predictions, labels_space):
    # Calculate and print classification report
    all_labels_np = all_labels.cpu().numpy()
    all_predictions_np = all_predictions.cpu().numpy()
    report = classification_report(all_labels_np, all_predictions_np, target_names=labels_space, digits=4)
    print("Classification Report:\n", report)

    # Calculate and print confusion matrix
    conf_matrix = confusion_matrix(all_labels_np, all_predictions_np)
    print("Confusion Matrix:\n", conf_matrix)

    # Calculate accuracy per class
    accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    for i, class_name in enumerate(labels_space):
        print(f"Accuracy for class '{class_name}': {accuracy_per_class[i]:.4f}")