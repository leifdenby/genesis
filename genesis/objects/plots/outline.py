if __name__ == "__main__": # noqa
    import matplotlib
    matplotlib.use("Agg")

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def cumsum_cutoff(v, frac=0.9):
    s = np.cumsum(v)
    n = np.argmin(np.abs(s - frac * np.max(s)))
    return n


def plot_outline(da, lx=10e3, frac=0.9):
    """
    Plot y-projected outline of objects which cumulatively contribute `frac` of
    the total volume of objects, splitting into separate subplots with distance
    `lx` along the x-axis
    """
    idx, idx_counts = np.unique(da, return_counts=True)
    idx = idx[1:]
    idx_counts = idx_counts[1:]
    order = np.argsort(idx_counts)[::-1]
    idx_by_size = idx[order]

    x_min, x_max = da.xt.min(), da.xt.max()

    windows = [dict(xt=slice(x0, x0 + lx)) for x0 in np.arange(x_min, x_max, lx)]

    print("Splitting outline plot into {} subplots".format(len(windows)))

    fig, axes = plt.subplots(nrows=len(windows), figsize=(14, 4))

    idxs_window = idx_by_size[: cumsum_cutoff(idx_counts[order], frac=frac)]

    for window, ax in zip(windows, axes):
        da_ = da.sel(**window)
        y_min, y_max = da_.yt.min(), da_.yt.max()

        for n, i in enumerate(tqdm.tqdm(idxs_window)):
            obj_mask = da_.where(da_ == i, other=0)
            y_mean = obj_mask.yt.where(obj_mask).mean()

            d = (y_mean - y_min) / (y_max - y_min)
            obj_mask.max(dim="yt").plot.contour(
                y="zt", ax=ax, add_colorbar=False, levels=[0.5], alpha=d
            )

        ax.set_aspect(1)


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument("objects_filename")
    argparser.add_argument("--frac", default=0.9, type=float)
    argparser.add_argument("--lx", default=10e3, type=float)

    args = argparser.parse_args()

    da = xr.open_dataarray(args.objects_filename, decode_times=False)

    plot_outline(da=da, lx=args.lx, frac=args.frac)

    fn_out = args.objects_filename.replace(".nc", ".outlines.png")
    plt.savefig(fn_out)
    print("Saved plot to {}".format(fn_out))
