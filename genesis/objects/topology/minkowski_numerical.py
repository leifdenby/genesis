# coding: utf-8

# In[1]:
import itertools
import os

import tqdm
import numpy as np
import xarray as xr
import seaborn as sns
from collections import OrderedDict
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize

from . import minkowski
from .plots.filamentarity_planarity import plot_reference as plot_fp_ref


# dx = 25.
# grid = make_grid(dx=dx)

# h = np.linspace(0, 1000)

# h0 = 1000.
# ls = 500


def make_grid(dx, lx=3e3, lz=2e3):
    ly = lx
    nx, ny, nz = int(lx / dx) + 1, int(ly / dx) + 1, int(lz / dx) + 1
    x = np.linspace(-lx / 2.0, lx / 2.0, nx)
    y = np.linspace(-ly / 2.0, ly / 2.0, ny)
    z = np.linspace(0, lz, nz)

    grid = xr.Dataset(coords=OrderedDict(x=x, y=y, z=z))

    return grid


# shearing function
f_shear = lambda ls, h0: lambda h: ls / h0 ** 2.0 * h ** 2


def len_fn_approx(h, ls, h0):
    "approximate distance along length"
    return np.sqrt(h ** 2.0 + f_shear(ls, h0)(h) ** 2.0)


def len_fn(h, ls, h0):
    "numerically integrated distance along length"
    # ax^2 + bc + c
    dldh = lambda h_: np.sqrt(1.0 + (ls / h0 ** 2.0 * 2.0 * h_) ** 2.0)
    return scipy.integrate.quad(dldh, 0, h)[0]


def find_scaling(ls, h0):
    """find height fraction `alpha` at which the top of the shape 
    is sheared a horizontal distance `ls` while keeping the length
    of the shape constant"""

    def fn(alpha):
        return h0 - len_fn(alpha * h0, ls=ls, h0=h0)

    return scipy.optimize.brentq(fn, 0.5, 1.0)


def make_plume_mask(grid, r0, h, shear_distance=0.0):
    """
    Return a dataset with a synthetic plume mask, `r0` denotes the characteristic radius,
    `h` the height and `shear_distance` the vertical sheared distance of the plume at
    height `h`
    """
    if h > grid.z.max():
        raise Exception(
            "Grid too small to contain plume, please increase z coordinate max"
        )

    ds = grid.copy()

    a = shear_distance / h ** 2
    ds["x_c"] = 0.0 * ds.z + a * ds.z ** 2.0
    ds["y_c"] = 0.0 * ds.z
    ds["xy_dist"] = np.sqrt((ds.x - ds.x_c) ** 2.0 + (ds.y - ds.y_c) ** 2.0)
    ds["r"] = r0 + 0.0 * ds.z
    ds["mask"] = ds.xy_dist < ds.r

    s = find_scaling(ls=shear_distance, h0=h)
    ds["mask"] = ds.mask.where(ds.z < h * s, False)

    ds["type"] = "thermal"

    # ensure coordinates are in correct order
    ds = ds.transpose("x", "y", "z")

    # cloud identification code isn't so good with objects that touch domain edge...
    ds["mask"].values[:, :, :1] = False
    ds["mask"].values[:, :, -1:] = False

    return ds


def make_thermal_mask(grid, r0, h, z_offset=0.0, shear_distance=0.0):
    """
    Return a dataset with a synthetic thermal mask, `r0` denotes the characteristic radius,
    `h` the height and `shear_distance` the vertical sheared distance of the thermal at
    height `h`
    """
    ds = grid.copy()

    s = find_scaling(ls=shear_distance, h0=h)

    z_c = s * h / 2.0 + z_offset

    a = shear_distance / h ** 2
    ds["x_c"] = 0.0 * ds.z + a * ds.z ** 2.0
    ds["y_c"] = 0.0 * ds.z
    ds["xy_dist"] = np.sqrt((ds.x - ds.x_c) ** 2.0 + (ds.y - ds.y_c) ** 2.0)
    ds["z_dist"] = np.abs(ds.z - z_c)

    ds["mask"] = (ds.xy_dist / r0) ** 2.0 + (ds.z_dist / (h * s / 2.0)) ** 2.0 < 1.0
    ds["type"] = "plume"

    # ensure coordinates are in correct order
    ds = ds.transpose("x", "y", "z")

    return ds


def make_mask(h, l, dx, shape, l_shear, with_plot):
    r0 = h / 2.0 / l

    # ensure the domain contains the full object
    lz = 2.5 * r0 * l
    lx_shear = 2 * l_shear
    lx_noshear = 2.5 * r0
    grid = make_grid(dx=dx, lx=lx_noshear + lx_shear, lz=lz)

    if l_shear != np.inf:
        grid = grid.sel(x=slice(-lx_noshear, None))

    if shape == "plume":
        ds = make_plume_mask(grid, r0=r0, h=h, shear_distance=l_shear,)
    elif shape == "thermal":
        ds = make_thermal_mask(grid, r0=r0, h=h, shear_distance=l_shear)

    if with_plot:
        ds.sel(y=0, method="nearest").mask.plot(y="z")
        plt.gca().set_aspect(1)

    fn_plot = "{}_{}_{}_{}_{}.png".format(h, l, dx, l_shear, shape)
    if not os.path.exists(fn_plot):
        fig, ax = plt.subplots()
        m = ds.sel(y=0, method="nearest").mask
        m.where(m, other=np.nan).plot(y="z", ax=ax, add_colorbar=False, cmap="Greys_r")
        plt.gca().set_aspect(1)
        plt.axis("off")
        plt.title("")
        plt.savefig(fn_plot, transparent=True)
        print("Wrote {}".format(fn_plot))

    return ds


def calc_scales(h, l, dx, shape, l_shear=0.0, with_plot=False):
    r0 = h / 2.0 / l

    ds = make_mask(h=h, l=l, dx=dx, shape=shape, l_shear=l_shear, with_plot=with_plot)
    # only one object so can cast mask to int
    object_labels = ds.mask.values.astype(int)

    scales = minkowski.calc_scales(object_labels=object_labels, dx=dx)

    scales["r0"] = r0

    return scales


def apply_all(ds, fn, dims=None):
    """
    Use coordinate dims in ds to provide arguments to fn
    """
    if dims is None:
        dims = ds.coords.keys()

    args = list(itertools.product(*[ds[d].values for d in dims]))

    def process(**kwargs):
        da = fn(**kwargs)
        if da is not None:
            for k, v in kwargs.items():
                da.coords[k] = v
            da = da.expand_dims(list(kwargs.keys()))
        return da

    data = [process(**dict(zip(dims, a))) for a in tqdm.tqdm(args)]

    if all([da is None for da in data]):
        return None
    else:
        return xr.merge(data).squeeze()


# # no-shear plume vs thermal

# In[33]:


def mask_plot_example():
    """
    Create a grid plot of mask for objects with change shear and thickness, but
    with constant length
    """
    import matplotlib.gridspec

    ds_study = xr.Dataset(
        coords=dict(
            h=[1000.0,],
            l=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            l_shear=[0.0, 500.0, 1000.0],  # [600., 400., 200., 0.],
            dx=[4.0,],
            shape=["thermal",],
        )
    )

    fig, axes = plt.subplots(
        nrows=len(ds_study.l_shear),
        ncols=len(ds_study.l),
        sharex=True,
        sharey=True,
        figsize=(9, 6),
    )

    def plot_mask(h, l, dx, shape, l_shear, with_plot=False):
        r0 = h / 2.0 / l

        ds = make_mask(
            h=h, l=l, dx=dx, shape=shape, l_shear=l_shear, with_plot=with_plot
        )

        ii = ds_study.l_shear.values.tolist().index(l_shear)
        jj = ds_study.l.values.tolist().index(l)

        ax = axes[ii, jj]
        # only one object so can cast mask to int
        im = ds.sel(y=0, method="nearest").mask.plot.pcolormesh(
            ax=ax, y="z", cmap="Greys", linewidth=0, rasterized=True,
        )
        # d = ds.sel(y=0, method='nearest').mask
        # ax.pcolormesh(d.x.values, d.z.values, d.values, alpha=0.1, cmap='Greys', linewidth=0, rasterized=True)
        im.colorbar.remove()
        plt.gca().set_aspect(1)
        ax.set_title(r"$l_s$={},$\lambda$={}".format(l_shear, int(l)))

    apply_all(ds_study, plot_mask)
    sns.despine()
    axes[0, 0].set_xlim(-300, 1000)
    plt.tight_layout()

    fn = "thermal-examples-shear.png"
    plt.savefig(fn, dpi=400)
    print("Wrote {}".format(fn))


def example2():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect(1)
    plot_fp_ref(ax=ax, shape="spheroid")

    ds_study = xr.Dataset(
        coords=dict(
            h=[1000.0,],
            l=[2.0, 3.0,],  # 4., 5., 6., 7., 8.],
            l_shear=[0.0, 500.0,],  # 1000.], #[600., 400., 200., 0.],
            dx=[4.0,],
            shape=["thermal",],
        )
    )
    ds_output = apply_all(ds_study, calc_scales)

    def format_length(v):
        if v == np.inf:
            return r"$\infty$"
        else:
            return r"{}m".format(v)

    try:
        for l_shear in list(ds_output.l_shear.values):
            ds_ = ds_output.sel(l_shear=l_shear).swap_dims(dict(l="planarity"))
            (l,) = ds_.filamentarity.plot.line(
                marker="s",
                markersize=4,
                linestyle="--",
                label=r"$l_s$={}".format(format_length(l_shear)),
                ax=ax,
            )
    except:
        ds_ = ds_output.swap_dims(dict(l="planarity"))
        ds_.filamentarity.plot.line(
            marker="s", markersize=4, linestyle="", ax=ax, label=ds_.shape.values
        )

    plt.xlim(-0.01, 0.2)
    plt.ylim(-0.01, 0.4)
    fig.legend()

    fn = "fp-plot-numerical.png"
    fig.savefig(fn, dpi=400)
    print("Wrote {}".format(fn))


def example3(output_fn, reference_shape='ellipsoid'):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect(1)
    plot_fp_ref(ax=ax, shape=reference_shape, lm_range=slice(1.0 / 4.0, 9), calc_kwargs=dict(N_points=400))

    ds_study = xr.Dataset(
        coords=dict(
            h=[1000.0,],
            l=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            l_shear=[0.0, 500.0, 1000.0, 1500.0],  # [600., 400., 200., 0.],
            dx=[4.0,],
            shape=["thermal",],
        )
    )
    ds_output = apply_all(ds_study, calc_scales)

    def format_length(v):
        if v == np.inf:
            return r"$\infty$"
        else:
            return r"{}m".format(v)

    # create a new flattened index
    ds_flat = ds_output.stack(i=ds_output.dims).reset_index("i")

    for i in range(len(ds_flat.i)):
        ds_ = ds_flat.isel(i=i)

        h, l, dx, shape = ds_.h.values, ds_.l.values, ds_.dx.values, ds_.shape.values
        l_shear = ds_.l_shear.values
        fn_img = "{}_{}_{}_{}_{}.png".format(h, l, dx, l_shear, shape)
        img = plt.imread(fn_img)

        lx, ly = 0.04, 0.04
        extent = [
            ds_.planarity - lx / 2.0,
            ds_.planarity + lx / 2.0,
            ds_.filamentarity - ly / 2.0,
            ds_.filamentarity + ly / 2.0,
        ]
        ax.imshow(img, extent=extent)

        # l, = ds_.filamentarity.plot(marker='s', markersize=4, linestyle='--',
        # label=r"$l_s$={}".format(format_length(l_shear)),
        # ax=ax)

    ax.set_xlim(-0.01, 0.25)
    ax.set_ylim(-0.01, 0.55)
    ax.set_aspect(0.5)
    sns.despine()
    ax.legend(loc='upper right')

    fig.savefig(output_fn, dpi=400)
    print("Wrote {}".format(output_fn))


if __name__ == "__main__":
    # example_with_synthetic_structures()
    example3()
