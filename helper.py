import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def plot_sample_data(XF,XW,YW,path="includes/sample-data-animation.mp4"):
    plt.clf()
    fig = plt.figure(figsize=(10, 10), dpi=100)

    plt.ion()

    plt.scatter(XF[:, 0], XF[:, 1], marker="s", s=7, color="lightgreen")

    scat = plt.scatter(
        XW[:, 0], XW[:, 1], marker="s", s=20, c=[(0, 0, 0, 1)] * len(XW)
    )
    label = plt.text(0, 0, '', fontsize=12)

    colors = []
    for obs in YW:
        colors.append([min(1 - abs(x) / 15, 1) for x in obs])

    colors = np.array(colors)

    def update_plot(i, scat):
        scat.set_array(colors[i])
        label.set_text(["Sp", "Su", "Fa", "Wi"][i % 4])
        return scat,

    anim = animation.FuncAnimation(
        fig, update_plot, frames=range(len(XW)), fargs=(scat,), interval=1000
    )
    
    plt.gray()

    # The following command requires ffmpeg to be installed on the system
    anim.save(path, fps=1)

    plt.close()