import math
import torch
import numpy as np
import pandas as pd

def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    dist = torch.clamp(dist, 0.0, np.inf)

    return dist


def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        ).transpose()
        site_stats[site_name] = describe[
            ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
        ]
    return site_stats


def visualize_posterior(samples):
    sites = list(samples.keys())

    r = int(math.ceil(math.sqrt(len(samples))))
    fig, axs = plt.subplots(nrows=r, ncols=r, figsize=(15, 13))
    fig.suptitle("Marginal Posterior Density", fontsize=16)

    for i, ax in enumerate(axs.reshape(-1)):
        if i >= len(sites):
            break
        site = sites[i]
        sns.distplot(samples[site], ax=ax)
        ax.set_title(site)

    handles, labels = ax.get_legend_handles_labels()
