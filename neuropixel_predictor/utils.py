def check_hyperparam_for_layers(hyperparameter, layers):
    if isinstance(hyperparameter, (list, tuple)):
        assert (
            len(hyperparameter) == layers
        ), f"Hyperparameter list should have same length {len(hyperparameter)} as layers {layers}"
        return hyperparameter
    elif isinstance(hyperparameter, int):
        return (hyperparameter,) * layers
