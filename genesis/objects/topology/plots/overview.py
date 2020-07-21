# coding: utf-8
if __name__ == "__main__":  # noqa
    import matplotlib

    matplotlib.use("Agg")


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from genesis.objects import get_data

from genesis.objects.topology.plots import filamentarity_planarity

from ....utils import wrap_angles


def ecd(d, **kwargs):
    y = np.sort(d)
    x = np.arange(1.0, len(y) + 1) / len(y)

    plt.plot(x, y, **kwargs)


def p_sum(x, y, frac=None, reverse=True, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    idx = np.argsort(x.compute().values)

    x_ = x.compute().values[idx]
    y_ = np.cumsum(y.compute().values[idx])

    plt.plot(x_, y_, **kwargs)

    plt.xlabel("{} [{}]".format(x.longname, x.units))
    plt.ylabel("cum sum of\n{} [{}]".format(y.longname, y.units))

    if frac is not None:
        y_lim = (1.0 - frac) * np.nanmax(y_)
        plt.axhline(y_lim, linestyle="--")

        return x_[np.nanargmin(np.abs(y_ - y_lim))]


def _add_unit_line_to_pairgrid(g):
    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            if i == j:
                continue

            ax = g.axes[i][j]

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            pmin = np.max([xmin, ymin])
            pmax = np.min([xmax, ymax])

            ax.plot([pmin, pmax], [pmin, pmax], linestyle="-", color="red")


def _plot_dist(x, dx, **kwargs):
    x_min = (x.min() / dx).astype(int) * dx
    x_max = (x.max() / dx).astype(int) * dx
    nbins = int((x_max - x_min) / dx)
    hist_kws = dict(range=(x_min, x_max))
    kwargs["bins"] = nbins
    sns.distplot(x, norm_hist=True, hist_kws=hist_kws, **kwargs)


def main(ds, variables, as_pairgrid=False, sharex=False):
    hue_label = "dataset"

    if as_pairgrid:
        df = pd.DataFrame(ds.to_dataframe().to_records()).dropna()
        g = sns.pairplot(df, hue=hue_label, vars=variables)
    else:
        df = pd.DataFrame(ds.to_dataframe().to_records()).dropna()
        g = sns.PairGrid(df, x_vars=variables, y_vars=variables[:1], hue=hue_label)

        def unlink_axes(_, __, **kwargs):
            ax = plt.gca()
            ax.get_shared_y_axes().remove(ax)
            ax.clear()

        def plot_var(x, _, **kwargs):
            if ds[x.name].units == "m":
                _plot_dist(x, dx=25.0, **kwargs)
            elif ds[x.name].units == "rad":
                x = wrap_angles(x)
                x = np.rad2deg(x)
                _plot_dist(x, dx=10.0, **kwargs)
            else:
                raise NotImplementedError(ds[x.name])

        g.map(plot_var)

        def scale_axes(x, _, **kwargs):
            if ds[x.name].units in [
                "m",
            ]:
                ax = plt.gca()
                x_c = x.mean()
                x_std = x.std()
                if x.min() < 0.0:
                    ax.set_xlim(x_c - x_std, x_c + x_std)
                else:
                    ax.set_xlim(0.0, x_std * 4.0)

        g.map(scale_axes)

        g.add_legend()
        # fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

        # for n, v in enumerate(variables):
        # ax = axes[n]
        # if hue_label:
        # ds_ = ds.where(ds[hue_label], drop=True)
        # ds_[v].plot.hist(ax=ax, bins=bins)
        # ax.set_xlim(0, None)
        # ax.set_title("")
    sns.despine()

    if sharex:
        [ax.set_xlim(0, ds.length.max()) for ax in g.axes.flatten()]


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument("base_name")
    argparser.add_argument("--objects", default="*")
    argparser.add_argument("--frac", default=0.9)

    args = argparser.parse_args()

    base_name = args.base_name
    frac = args.frac

    ds = get_data(base_name=base_name, mask_identifier=args.objects)

    if "r_equiv" in ds.data_vars:
        plt.figure()
        r_lim = p_sum(ds.r_equiv, ds.q_flux__integral, marker=".", frac=frac)

        fn_out = "{}.minkowski_scales1.png".format(base_name)
        plt.savefig(fn_out)
        print("Saved plot to {}".format(fn_out))

        plt.figure()
        ds_ = ds.where(ds.r_equiv > r_lim, drop=True)
        g = sns.jointplot(ds_.q_flux__integral, ds_.zt__maximum)

        fn_out = "{}.minkowski_scales2.png".format(base_name)
        plt.savefig(fn_out)
        print("Saved plot to {}".format(fn_out))
    else:
        print("volume integral var missing skipping r_equiv plot")

    filamentarity_planarity.fp_plot(ds=ds)
    objects_s = ["all", args.objects][args.objects != "*"]
    fn_out = "{}.{}.filamentarity.vs.planarity.pdf".format(args.base_name, objects_s)
    plt.savefig(fn_out)
    print("Saved filamentarity planarity plot to {}".format(fn_out))
