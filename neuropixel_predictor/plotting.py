import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from neuropixel_predictor.mei.generate import plot_mei

TIMESERIES_SAVE_DIR = 'timeseries'

def plot_all_methods_per_neuron(
    training_Y,
    training_Yhat,
    training_ids,
    test_Y,
    test_Yhat,
    test_ids,
    reconstruction_metadata,
    meis,
    meis_steps='',
    run_name='',
    cluster_ids=None,
    training_neuron_stats=None,
    test_neuron_stats=None,
    save_dir=None,
):
    date = datetime.now().strftime("%Y-%m-%d")
    data_keys = list(training_Y.keys())

    for dk in data_keys:
        N, _ = training_Y[dk].shape

        for n in range(N):
            fig, axes = plt.subplots(
                3, 1,
                figsize=(25, 15),
                constrained_layout=True,
            )

            # 1️⃣ Test/Validation reconstruction plot
            plot_reconstruction_single_neuron(
                Y=test_Y,
                Yhat=test_Yhat,
                ids=test_ids,
                reconstruction_metadata=reconstruction_metadata,
                dk=dk,
                neuron_idx=n,
                ax=axes[0],
                cluster_ids=cluster_ids,
                neuron_wise_stats=test_neuron_stats,
                training_or_test='Test'
            )

            # 2️⃣ Training Reconstruction plot
            plot_reconstruction_single_neuron(
                Y=training_Y,
                Yhat=training_Yhat,
                ids=training_ids,
                reconstruction_metadata=reconstruction_metadata,
                dk=dk,
                neuron_idx=n,
                ax=axes[1],
                cluster_ids=cluster_ids,
                neuron_wise_stats=training_neuron_stats,
                training_or_test='Training'
            )

            # 3️⃣ MEI plot
            plot_mei(
                meis[dk][n],
                neuron_idx=n,
                title=f"MEI - Neuron: {n} (#{cluster_ids[dk][n]}), Steps: {meis_steps}",
                ax=axes[2]
            )

            fig.suptitle(
                f"({run_name}) - Neuron {n} (#{cluster_ids[dk][n]}) | Data key: {dk}",
                fontsize=18,
            )

            if save_dir is not None:
                sub_dir = f"{date}_({run_name})"
                out_dir = os.path.join(save_dir, sub_dir, dk)
                os.makedirs(out_dir, exist_ok=True)
                fname = f"neuron_{n:03d}.png"
                fig.savefig(
                    os.path.join(out_dir, fname),
                    dpi=150,
                    bbox_inches="tight",
                )

            plt.show()
            plt.close(fig)

def plot_reconstruction_single_neuron(
    Y,
    Yhat,
    ids,
    reconstruction_metadata,
    dk,
    neuron_idx,
    ax,
    training_or_test='',
    cluster_ids=None,
    neuron_wise_stats=None,
):
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def extract_stimulus(id_str):
        return id_str.split("_")[0]

    colors = plt.cm.tab10.colors
    stim_order = list(reconstruction_metadata.keys())

    Y_dk = to_numpy(Y[dk])        # (N, D)
    Yhat_dk = to_numpy(Yhat[dk])  # (N, D)
    ids_dk = ids[dk]

    stim_ids = np.array([extract_stimulus(i) for i in ids_dk])
    N, D = Y_dk.shape

    x_offset = 0
    legend_handles = []

    for si, stim in enumerate(stim_order):
        idx = np.where(stim_ids == stim)[0]
        if len(idx) == 0:
            continue

        color = colors[si % len(colors)]
        x = np.arange(x_offset, x_offset + len(idx))

        # True firing rate
        ax.plot(
            x,
            Y_dk[neuron_idx, idx],
            color=color,
            linewidth=1,
        )

        # Predicted firing rate
        ax.plot(
            x,
            Yhat_dk[neuron_idx, idx],
            color='red',
            # linestyle=":",
            linewidth=1,
        )

        legend_handles.append(
            plt.Line2D(
                [0], [0],
                color=color,
                linewidth=2,
                label=reconstruction_metadata[stim],
            )
        )

        x_offset += len(idx)

    legend_handles.append(
        plt.Line2D([0], [0], color="red", linestyle=":", linewidth=2, label="model prediction")
    )

    cluster = f"{neuron_idx} (#{cluster_ids[dk][neuron_idx]})" if cluster_ids else neuron_idx
    title = f"{training_or_test} - Neuron {cluster} | Data key: {dk}"
    if neuron_wise_stats is not None:
        title += f" | corrcoef: {neuron_wise_stats[dk][neuron_idx]:.4f}"

    ax.set_title(title)
    ax.set_xlabel(f"{training_or_test} Datapoint index")
    ax.set_ylabel("Firing rate")
    ax.set_xlim(0, D - 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(handles=legend_handles)


def plot_mei(mei_tensor, neuron_idx, title="MEI", save_folder=None, ax=None):
    """
    mei_tensor: torch.Tensor with shape (1, C, H, W) or (C, H, W)
    """
    # Move to CPU and convert to numpy
    img = mei_tensor.detach().cpu().squeeze()

    img = torch.moveaxis(img, 1, 0) # Flip the axis to have (H, W)

    if ax:
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
    else:
        plt.title(title)
        plt.axis("off")

    if save_folder:
        filename = f"mei_neuron{neuron_idx:03d}.png"
        filepath = os.path.join(save_folder, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.show()
        # plt.close(fig)
