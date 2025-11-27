import torch
import matplotlib.pyplot as plt

from .methods import MEIMethod, MEIConfig

def generate_mei(model, data_key, neuron_idx, image_shape,
                 steps=300, mode="cei", device="cuda"):
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

    mei = mei_method.optimize(model, data_key, neuron_idx, image_shape, steps=steps, device=device)
    return mei

def plot_mei(mei_tensor, title="MEI"):
    """
    mei_tensor: torch.Tensor with shape (1, C, H, W) or (C, H, W)
    """
    # Move to CPU and convert to numpy
    img = mei_tensor.detach().cpu().squeeze()

    # If single channel: shape = (H, W)
    if img.dim() == 2:
        plt.imshow(img, cmap='gray')

    # If multi-channel: shape = (C, H, W)
    elif img.dim() == 3:
        # reorder to (H, W, C)
        img = img.permute(1, 2, 0)
        plt.imshow(img)

    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    plt.title(title)
    plt.axis("off")
    plt.show()
