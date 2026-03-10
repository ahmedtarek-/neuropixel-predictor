import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class MEIConfig:
    """Configuration for MEI optimization."""
    lr: float = 0.05
    steps: int = 300
    init_std: float = 0.1
    l2_lambda: float = 1e-3
    tv_lambda: float = 5e-4
    blur_sigma_start: float = 5
    blur_sigma_end: float = 0.5
    clip_min: float = -5.0
    clip_max: float = 5.0
    mode: str = "cei"   # "cei", "vei_plus", "vei_minus"


def total_variation(x):
    """Total variation regularizer."""
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w

def gaussian_kernel2d(sigma, device="cpu", max_kernel_size=49):
    """
    Create a fixed-size 2D Gaussian kernel with sigma.
    Args:
        sigma: blur width
        max_kernel_size: odd integer (e.g., 49, 31)
    """
    # Ensure odd
    assert max_kernel_size % 2 == 1

    # Create coordinate grid
    ax = torch.arange(max_kernel_size, dtype=torch.float32, device=device) - (max_kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax)
    
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2 + 1e-8))
    kernel = kernel / kernel.sum()

    return kernel

def blur_image(img, sigma, device='cpu'):
    """
    Blur `img` with a fixed-size Gaussian kernel.
    """
    B, C, W, H = img.shape
    kernel = gaussian_kernel2d(sigma, device=device, max_kernel_size=49)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape = (1,1,K,K)
    kernel = kernel.repeat(C, 1, 1, 1)

    # same padding
    padding = (kernel.shape[-1] // 2, kernel.shape[-2] // 2)

    out = F.conv2d(img, kernel, padding=padding, groups=C)
    return out

def blur_schedule(step, steps, sigma_start, sigma_end):
    """
    Linearly interpolate blur sigma:
      starts at sigma_start, ends at sigma_end
    """
    t = step / (steps - 1)
    return sigma_start * (1 - t) + sigma_end * t


class MEIMethod:
    def __init__(self, config: MEIConfig):
        self.config = config

    def loss(self, model, image, data_key, idx):
        """Define the objective depending on CEI / VEI±."""
        if self.config.mode == "cei":
            pred = model(image, data_key=data_key)[0, 0, idx]
            loss = -pred

        elif self.config.mode == "vei_plus":
            pred_var = model.predict_variance(image, data_key=data_key)[0, 0, idx]
            loss = -pred_var

        elif self.config.mode == "vei_minus":
            pred_var = model.predict_variance(image, data_key=data_key)[0, 0, idx]
            loss = pred_var  # minimize variance

        else:
            raise ValueError(f"Unknown MEI mode: {self.config.mode}")

        # Regularizers
        l2_reg = self.config.l2_lambda * (image ** 2).mean()
        tv_reg = self.config.tv_lambda * total_variation(image)

        return loss + l2_reg + tv_reg

    def optimize(self, model, data_key, neuron_idx, image_shape, steps, device="cuda"):
        """Run gradient ascent to obtain MEI."""
        # Initialize input with Gaussian noise
        image = (
            torch.randn(image_shape, device=device) * self.config.init_std
        ).requires_grad_(True)

        optimizer = torch.optim.Adam([image], lr=self.config.lr)

        for step in range(steps):
            optimizer.zero_grad()

            loss = self.loss(model, image, data_key, neuron_idx)
            # print("loss: ", loss)
            loss.backward()

            # blur gradient before step
            sigma = blur_schedule(step, steps, self.config.blur_sigma_start, self.config.blur_sigma_end)  # adjust
            with torch.no_grad():
                image.grad.data = blur_image(image.grad.data, sigma, device=image.device)

            optimizer.step()

            # Keep image in allowed range
            image.data.clamp_(self.config.clip_min, self.config.clip_max)

        return image.detach()
