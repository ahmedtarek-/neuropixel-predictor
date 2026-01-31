import random
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

TIMESERIES_SAVE_DIR = 'timeseries'

def generate_reconstructed_data(test_dataloaders, model):
    # 1. Generate data per every neuron 
    Y = dict([[k, []] for k in test_dataloaders.keys()])
    Yhat = dict([[k, []] for k in test_dataloaders.keys()])
    ids = dict([[k, []] for k in test_dataloaders.keys()])

    model.eval()
    device = torch.device("mps")

    with torch.no_grad():
        for data_key in test_dataloaders.keys():
            for x, y, ids_batch in test_dataloaders[data_key]:
                x, y = x.to(device), y.to(device)
                # 1.2 Calculate model prediction
                yhat = model(x, data_key=data_key)
                yhat = yhat[:, 0, :]
                Y[data_key].append(y)
                Yhat[data_key].append(torch.exp(yhat))
                ids[data_key].append(ids_batch)

    # 2. Concatenate and Sort according to IDs.
    for dk in Y.keys():
        # 2.1 Concatenate
        Y[dk] = torch.cat(Y[dk], dim=0)
        Yhat[dk] = torch.cat(Yhat[dk], dim=0)
        ids[dk] = [str(i) for i in list(sum(ids[dk], ()))] # Flatten the list of tuples to one list

        # 2.2 Sort according to IDs
        numerical_sort = sorted(ids[dk], key=lambda i: int(i.split('_')[1]))
        sorted_ids = sorted(numerical_sort, key=lambda i: i.split('_')[0])
        sorting_indices = [sorted_ids.index(i) for i in ids[dk]]

        Y[dk] = Y[dk][sorting_indices].T
        Yhat[dk] = Yhat[dk][sorting_indices].T

    return Y, Yhat, ids

def plot_reconstructions_by_neuron(
    Y,
    Yhat,
    ids,
    reconstruction_metadata,
    cluster_ids=None,
    neuron_wise_stats=None
):
    date = datetime.now().strftime("%Y-%m-%d")
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def extract_stimulus(id_str):
        return id_str.split("_")[0]

    data_keys = list(Y.keys())
    stim_order = list(reconstruction_metadata.keys())
    colors = plt.cm.tab10.colors

    for dk in data_keys:
        Y_dk = to_numpy(Y[dk])        # (N, D)
        Yhat_dk = to_numpy(Yhat[dk])  # (N, D)
        ids_dk = ids[dk]

        stim_ids = np.array([extract_stimulus(i) for i in ids_dk])

        N, D = Y_dk.shape

        for n in range(N):
            fig, ax = plt.subplots(figsize=(14, 4))
            x_offset = 0
            legend_handles = []

            for si, stim in enumerate(stim_order):
                idx = np.where(stim_ids == stim)[0]
                if len(idx) == 0:
                    continue

                color = colors[si % len(colors)]
                x = np.arange(x_offset, x_offset + len(idx))

                ax.plot(
                    x,
                    Y_dk[n, idx],
                    color=color,
                    linewidth=1,
                )

                ax.plot(
                    x,
                    Yhat_dk[n, idx],
                    color='red',
                    # linestyle=":",
                    linewidth=1
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
                    plt.Line2D([0], [0], color='red', linewidth=2, label='model prediction')
                )

            cluster = f"{n} (#{cluster_ids[dk][n]})" if cluster_ids else n
            title = f"Neuron {cluster} | Data key: {dk}"
            if neuron_wise_stats:
                title += f" | corrcoef: {neuron_wise_stats[dk][n]:.4f}"
            ax.set_title(title)
            ax.set_xlabel("Datapoint index (stimulus concatenated)")
            ax.set_ylabel("Firing rate")
            ax.set_xlim(0, D - 1)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(handles=legend_handles)

            plt.tight_layout()
            # Save plot
            sub_dir = f"{date}_({run.name})"
            save_folder = os.path.join(TIMESERIES_SAVE_DIR, sub_dir, dk)
            os.makedirs(save_folder, exist_ok=True)
            filename = f"timeseries_neuron{n:03d}.png"
            filepath = os.path.join(save_folder, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            
            plt.show()

def plot_metric(history, history_metric, title, label):
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    colors = plt.cm.tab10.colors  # distinct colors

    for i, (dk, dk_history) in enumerate(history.items()):
        values = dk_history[history_metric]
        epochs = range(1, len(values) + 1)
        color = colors[i % len(colors)]

        final_value = values[-1]
        legend_label = f"{dk} (final={final_value:.4f})"

        ax.plot(
            epochs,
            values,
            label=legend_label,
            linewidth=2,
            color=color
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.show()

def calculate_corr_stats(test_dataloaders, model):
    # 1. Accumulate the Y_hats and Y
    Y = dict([[k, []] for k in test_dataloaders.keys()])
    Yhat = dict([[k, []] for k in test_dataloaders.keys()])
    corr_coef = {}

    model.eval()
    device = torch.device("mps")

    # 2. Assemble Y and Yhat
    with torch.no_grad():
        for data_key in test_dataloaders.keys():
            for x, y, id in test_dataloaders[data_key]:
                x, y = x.to(device), y.to(device)
                # 1.2 Calculate model prediction
                yhat = model(x, data_key=data_key)
                yhat = yhat[:, 0, :]
                Y[data_key].append(y)
                Yhat[data_key].append(yhat)

    # 3. Calculate correlation coefficient
    for dk in Y.keys():
        y = torch.cat(Y[dk], dim=0)
        yhat = torch.cat(Yhat[dk], dim=0)
        print(yhat.shape)

        num_neurons = yhat.shape[1]
        neuron_corrs = torch.empty(num_neurons)
        
        for n in range(num_neurons):
            x = yhat[:, n]
            z = y[:, n]

            # handle degenerate neurons safely
            if x.std() == 0 or z.std() == 0:
                neuron_corrs[n] = float('nan')
            else:
                neuron_corrs[n] = torch.corrcoef(
                    torch.stack([x, z])
                )[0, 1]

        corr_coef[dk] = neuron_corrs

    return corr_coef

def calculate_r2_stats(test_dataloaders, model):
    # 1.1 Remove neurons with zero variance
            # var = y.var(dim=0)
            # valid = var > 1e-6
            # y = y[:, valid]

    # 1. Accumulate the Y_hats and Y
    Y = dict([[k, []] for k in test_dataloaders.keys()])
    Yhat = dict([[k, []] for k in test_dataloaders.keys()])
    r2 = {}
    r2_stats = {}

    model.eval()
    device = torch.device("mps")

    with torch.no_grad():
        for data_key in test_dataloaders.keys():
            for x, y, id in test_dataloaders[data_key]:
                x, y = x.to(device), y.to(device)
                # 1.2 Calculate model prediction
                yhat = model(x, data_key=data_key)
                yhat = yhat[:, 0, :]
                Y[data_key].append(y)
                Yhat[data_key].append(yhat)

    for dk in Y.keys():
        y = torch.cat(Y[dk], dim=0)
        yhat = torch.cat(Yhat[dk], dim=0)
        
        # 3. Calculate R2
        ss_res = ((y - yhat) ** 2).sum(dim=0)
        ss_tot = ((y - yhat.mean(dim=0)) ** 2).sum(dim=0)
        r2[dk] = 1 - ss_res / ss_tot
        r2[dk] = r2[dk].detach().cpu().numpy()

        r2_stats[dk] = {
            "median": format(np.median(r2[dk]), '.4f'),
            "90th_percentile": format(np.percentile(r2[dk], 90), '.4f'),
            "fraction_r2_gt_0": format(np.mean(r2[dk] > 0), '.4f'),
            "units_r2_gt_0.1": [int(i) for i in (r2[dk] > 0.1).nonzero()[0]]
        }
        # 4. Calculate statistics
        print("\nFor DataKey: ", dk)
        print("Median R²:", r2_stats[dk]['median'])
        print("90th percentile:", r2_stats[dk]['90th_percentile'])
        print("Fraction R² > 0:", r2_stats[dk]['fraction_r2_gt_0'])
        print("Neurons with high R²: ",  r2_stats[dk]['units_r2_gt_0.1'])

    return r2_stats

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
