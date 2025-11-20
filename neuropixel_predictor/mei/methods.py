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
    clip_min: float = -5.0
    clip_max: float = 5.0
    mode: str = "cei"   # "cei", "vei_plus", "vei_minus"


def total_variation(x):
    """Total variation regularizer."""
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w


class MEIMethod:
    def __init__(self, config: MEIConfig):
        self.config = config

    def loss(self, model, image, idx):
        """Define the objective depending on CEI / VEI±."""
        if self.config.mode == "cei":
            pred = model(image)[0, 0, idx]
            loss = -pred

        elif self.config.mode == "vei_plus":
            pred_var = model.predict_variance(image)[0, 0, idx]
            loss = -pred_var

        elif self.config.mode == "vei_minus":
            pred_var = model.predict_variance(image)[0, 0, idx]
            loss = pred_var  # minimize variance

        else:
            raise ValueError(f"Unknown MEI mode: {self.config.mode}")

        # Regularizers
        l2_reg = self.config.l2_lambda * (image ** 2).mean()
        tv_reg = self.config.tv_lambda * total_variation(image)

        return loss + l2_reg + tv_reg

    def optimize(self, model, neuron_idx, image_shape, steps, device="cuda"):
        """Run gradient ascent to obtain MEI."""
        # Initialize input with Gaussian noise
        image = (
            torch.randn(image_shape, device=device) * self.config.init_std
        ).requires_grad_(True)

        optimizer = torch.optim.Adam([image], lr=self.config.lr)

        for step in range(steps):
            optimizer.zero_grad()

            loss = self.loss(model, image, neuron_idx)
            print("loss: ", loss)
            loss.backward()

            optimizer.step()

            # Keep image in allowed range
            image.data.clamp_(self.config.clip_min, self.config.clip_max)

        return image.detach()
