import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns
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


def plot_data(XF, XW, YW, path="includes/animation.mp4"):
    plt.clf()
    fig = plt.figure(figsize=(10, 10), dpi=100)

    plt.ion()

    plt.scatter(XF[:, 0], XF[:, 1], marker="s", s=7, color="lightgreen")

    scat = plt.scatter(
        XW[:, 0], XW[:, 1], marker="s", s=20, c=[(0, 0, 0, 1)] * len(XW)
    )
    label = plt.text(0, 0, "", fontsize=12)

    colors = []
    for obs in YW:
        colors.append([min(1 - abs(x) / 15, 1) for x in obs])

    colors = np.array(colors)

    def update_plot(i, scat):
        scat.set_array(colors[i])
        label.set_text(["Sp", "Su", "Fa", "Wi"][i % 4])
        return (scat,)

    anim = animation.FuncAnimation(
        fig, update_plot, frames=range(len(XW)), fargs=(scat,), interval=1000
    )

    plt.gray()

    # The following command requires ffmpeg to be installed on the system
    anim.save(path, fps=1)

    plt.close()


def plot_2d_dist(param1, param2, samples, model, sci=True):
    plt.rcParams.update({"font.size": 15})

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(samples[param1], samples[param2])
    ax.set(
        xlabel=r"Value of $\{}$".format(param1),
        ylabel=r"Value of $\{}$".format(param2),
        title="Scatter Plot of Parameter Samples from Posterior Distribution",
    )
    ax.set_xlim((samples[param1].min(), samples[param1].max()))
    ax.set_ylim((samples[param2].min(), samples[param2].max()))
    if sci:
        ax.ticklabel_format(style='sci', scilimits=(-3, 2))

    plt.savefig(
        "includes/model{}/scatter_{}_{}.png".format(model, param1, param2)
    )
    plt.close()

    # pcolormesh
    fig, ax = plt.subplots(figsize=(10, 6))
    hist_2d = ax.hist2d(
        samples[param1], samples[param2], bins=(15, 15), cmap=plt.cm.jet
    )
    ax.set(
        xlabel=r"Value of $\{}$".format(param1),
        ylabel=r"Value of $\{}$".format(param2),
        title="2d Histogram of Parameter Samples from Posterior Distribution",
    )
    if sci:
        ax.ticklabel_format(style='sci', scilimits=(-3, 2))

    plt.colorbar(hist_2d[3], ax=ax)

    plt.savefig(
        "includes/model{}/hist_{}_{}.png".format(model, param1, param2)
    )
    plt.close()


def plot_timeline(variable, samples, path, xticks=None):
    # Grab quantiles and medians and plot them
    medians = np.median(samples, 0)
    quantiles = np.quantile(
        samples, q=(0.025, 0.975), axis=0
    )

    fig, ax = plt.subplots(figsize=(10, 6),)
    ax.plot(medians)
    ax.set(
        ylabel=r"Posteriror Values of $\{}$".format(variable),
        xlabel=r"Season",
        title=r"Scatter Plot of Median $\{}$ Values".format(variable),
    )
    if xticks is not None:
        plt.xticks(np.arange(len(xticks)), xticks)

    plotline, caplines, barlinecols = plt.errorbar(
        list(range(len(medians))),
        medians,
        yerr=quantiles[1],
        lolims=True,
        label="uplims=True",
        color="orange",
    )
    caplines[0].set_marker("_")
    caplines[0].set_markersize(10)

    plotline2, caplines2, barlinecols2 = plt.errorbar(
        list(range(len(medians))),
        medians,
        yerr=quantiles[0],
        uplims=True,
        label="uplims=True",
        color="orange",
    )
    caplines2[0].set_marker("_")
    caplines2[0].set_markersize(10)

    plt.savefig(path)
    plt.close()
