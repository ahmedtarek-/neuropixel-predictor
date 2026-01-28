import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt

from .methods import MEIMethod, MEIConfig

IMAGE_WIDTH = 36
IMAGE_HEIGHT = 22

SAVE_DIR = 'meis'

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

def plot_mei(mei_tensor, neuron_idx, title="MEI", save_folder=None):
    """
    mei_tensor: torch.Tensor with shape (1, C, H, W) or (C, H, W)
    """
    # Move to CPU and convert to numpy
    img = mei_tensor.detach().cpu().squeeze()

    img = torch.moveaxis(img, 1, 0) # Flip the axis to have (H, W)

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

    if save_folder:
        filename = f"mei_neuron{neuron_idx:03d}.png"
        filepath = os.path.join(save_folder, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.show()
        # plt.close(fig)

def generate_and_save_meis(n_neurons_dict, steps=600, folder_suffix=None, cluster_ids=None, title_suffix=""):
    date = datetime.now().strftime("%Y-%m-%d")

    sub_dir = f"{date}_({folder_suffix})" if folder_suffix else date
    os.makedirs(SAVE_DIR, exist_ok=True)

    image_shape = (1, 1, IMAGE_WIDTH, IMAGE_HEIGHT)
    
    total_meis = sum(n_neurons_dict.values())
    current_mei = 0
    
    print(f"Generating {total_meis} MEIs across {len(n_neurons_dict.keys())} data keys...")
    
    for data_key in n_neurons_dict.keys():
        num_neurons = n_neurons_dict[data_key]
        
        # Create required directories
        key_save_dir = os.path.join(SAVE_DIR, sub_dir, data_key)
        os.makedirs(key_save_dir, exist_ok=True)

        print(f"\nProcessing data_key: {data_key} ({num_neurons} neurons)")
        for neuron_idx in range(num_neurons):
            current_mei += 1

            # Generate MEI
            mei = generate_mei(
                model,
                data_key,
                neuron_idx,
                image_shape,
                steps=steps,
                mode='cei',
                device=device
            )
            # Create figure and plot
            fig, ax = plt.subplots(figsize=(8, 6))

            cluster = f"{neuron_idx} (#{cluster_ids[data_key][neuron_idx]})" if cluster_ids else neuron_idx
            plot_mei(
                mei,
                neuron_idx,
                title=f"MEI - {data_key}, Neuron: {cluster}, Steps: {steps} {title_suffix}",
                save_folder=key_save_dir
            )
