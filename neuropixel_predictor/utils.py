import random
import torch
import numpy as np

def check_hyperparam_for_layers(hyperparameter, layers):
    if isinstance(hyperparameter, (list, tuple)):
        assert (
            len(hyperparameter) == layers
        ), f"Hyperparameter list should have same length {len(hyperparameter)} as layers {layers}"
        return hyperparameter
    elif isinstance(hyperparameter, int):
        return (hyperparameter,) * layers


def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
