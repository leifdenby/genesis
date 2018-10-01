
# coding: utf-8

# In[1]:

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from genesis.objects import get_data


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


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument('base_name')
    argparser.add_argument('--frac', default=0.9)

    args = argparser.parse_args()

    base_name = args.base_name
    frac = args.frac

    ds = get_data(base_name=base_name)

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
