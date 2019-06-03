
# coding: utf-8

# In[1]:

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

from genesis.objects import get_data

from genesis.objects.topology.plots import filamentarity_planarity


def ecd(d, **kwargs):
    y = np.sort(d)
    x = np.arange(1.0, len(y)+1)/len(y)
    
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
        y_lim = (1.0 - frac)*np.nanmax(y_)
        plt.axhline(y_lim, linestyle='--')
        
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

            ax.plot([pmin, pmax], [pmin, pmax], linestyle='-', color='red')


def main(ds, variables, as_pairgrid=False, sharex=False):
    # N_objects_orig = int(ds.object_id.count())
    # ds = ds.dropna('object_id')
    # N_objects_nonan = int(ds.object_id.count())
    # print("{} objects out of {} remain after ones with nan for length, width"
          # " or thickness have been remove".format(N_objects_nonan,
          # N_objects_orig))

    hue_label = 'dataset'

    if as_pairgrid:
        df = pd.DataFrame(ds.to_dataframe().to_records()).dropna()
        g = sns.pairplot(df, hue=hue_label, vars=variables)
    else:
        fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

        for n, v in enumerate(variables):
            ax = axes[n]
            if not exclude_thin:
                _, bins, _ = ds[v].plot.hist(ax=ax)
            else:
                bins = None
            if hue_label:
                ds_ = ds.where(ds[hue_label], drop=True)
                ds_[v].plot.hist(ax=ax, bins=bins)
            ax.set_xlim(0, None)
            ax.set_title("")
    sns.despine()

    if sharex:
        [ax.set_xlim(0, ds.length.max()) for ax in g.axes.flatten()]



if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument('base_name')
    argparser.add_argument('--objects', default='*')
    argparser.add_argument('--frac', default=0.9)

    args = argparser.parse_args()

    base_name = args.base_name
    frac = args.frac

    ds = get_data(base_name=base_name, mask_identifier=args.objects)

    if 'r_equiv' in ds.data_vars:
        plt.figure()
        r_lim = p_sum(ds.r_equiv, ds.q_flux__integral, marker='.', frac=frac)

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
