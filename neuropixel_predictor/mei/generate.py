from .methods import MEIMethod, MEIConfig

def generate_mei(model, neuron_idx, image_shape,
                 mode="cei", device="cuda"):
    """
    High-level wrapper to generate a MEI.
    """
    config = MEIConfig(mode=mode)
    mei_method = MEIMethod(config)

    model.to(device)
    model.eval()

    with torch.no_grad():
        # Optional: warmup or model-specific initialization
        pass

    mei = mei_method.optimize(model, neuron_idx, image_shape, device=device)
    return mei
