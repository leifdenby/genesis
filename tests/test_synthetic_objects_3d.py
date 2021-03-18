# coding: utf-8
"""
Routines for creating masks of parameterised synthetic 3D shapes
"""

import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile

from genesis.utils.xarray import apply_all
from genesis.objects.synthetic import discrete, ellipsoid
from genesis.objects.topology.plots import synthetic_discrete_objects

TEMP_DATA_PATH = tempfile.TemporaryDirectory().name


def test_create_object_masks_3d():
    """
    Create a grid plot of mask for objects with change shear and thickness, but
    with constant length
    """
    ds_study = xr.Dataset(
        coords=dict(
            h=[1000.0],
            length=[
                2.0,
                3.0,
            ],
            l_shear=[
                0.0,
                500.0,
            ],
            dx=[25.0],
            shape=["thermal"],
        )
    )

    fig, axes = plt.subplots(
        nrows=len(ds_study.l_shear),
        ncols=len(ds_study.length),
        sharex=True,
        sharey=True,
        figsize=(9, 6),
    )

    def plot_mask(h, length, dx, shape, l_shear):
        ds = discrete.make_mask(
            h=h, length=length, dx=dx, shape=shape, l_shear=l_shear  # noqa,
        )

        ii = ds_study.l_shear.values.tolist().index(l_shear)
        jj = ds_study.length.values.tolist().index(length)

        ax = axes[ii, jj]
        # only one object so can cast mask to int
        im = ds.sel(y=0, method="nearest").mask.plot.pcolormesh(
            ax=ax,
            y="z",
            cmap="Greys",
            linewidth=0,
            rasterized=True,
        )
        # d = ds.sel(y=0, method='nearest').mask
        # ax.pcolormesh(d.x.values, d.z.values, d.values, alpha=0.1, cmap='Greys', linewidth=0, rasterized=True)
        im.colorbar.remove()
        plt.gca().set_aspect(1)
        ax.set_title(r"$l_s$={},$\lambda$={}".format(l_shear, int(length)))

    apply_all(ds_study, plot_mask)
    sns.despine()
    axes[0, 0].set_xlim(-300, 1000)
    plt.tight_layout()


def _make_test_grid():
    lx, ly, lz = 100, 100, 100

    x_ = np.arange(-lx / 2, lx / 2, 1)
    y_ = np.arange(-ly / 2, ly / 2, 1)
    z_ = np.arange(-lz / 2, lz / 2, 1)

    ds = xr.Dataset(coords=dict(x=x_, y=y_, z=z_))

    ds["x_3d"], ds["y_3d"], ds["z_3d"] = xr.broadcast(ds.x, ds.y, ds.z)
    ds.attrs["lx"] = lx
    ds.attrs["ly"] = ly
    ds.attrs["lz"] = lz

    return ds


def test_plot_shape_mask():
    ds = _make_test_grid()
    lx = ds.lx

    a, b = lx / 4.0, lx / 2.0
    ds["mask"] = ds.x ** 2.0 / a ** 2.0 + ds.y ** 2.0 / b ** 2.0 + ds.z ** 2.0 < 1.0

    ds.sel(z=0, method="nearest").mask.plot()
    plt.gca().set_aspect(1)

    a, b = lx / 4.0, lx / 2.0
    ds["mask"] = (
        ds.x_3d ** 2.0 / a ** 2.0 + ds.y_3d ** 2.0 / b ** 2.0 + ds.z_3d ** 2.0 < 1.0
    )

    ds.sel(z=0, method="nearest").mask.plot()
    plt.gca().set_aspect(1)


def test_make_ellipsoid_mask():
    ds = _make_test_grid()
    lx = ds.lx

    a = lx / 4.0
    b = lx / 2.0
    c = a
    mask = ellipsoid.make_mask(
        grid=(ds.x_3d, ds.y_3d, ds.z_3d), theta=0, phi=0, a=a, b=b, c=c
    )

    ds["mask_rot"] = (ds.x_3d.dims, mask)

    ellipsoid.plot_shape_mask(ds)

    mask = ellipsoid.make_mask(
        grid=(ds.x_3d, ds.y_3d, ds.z_3d), theta=45.0 / 180 * 3.14, phi=0, a=a, b=b, c=c
    )
    ds["mask_rot"] = (ds.x_3d.dims, mask)

    ellipsoid.plot_shape_mask(ds)

    mask = ellipsoid.make_mask(
        grid=(ds.x_3d, ds.y_3d, ds.z_3d), theta=0, phi=60.0 / 180 * 3.14, a=a, b=b, c=c
    )
    ds["mask_rot"] = (ds.x_3d.dims, mask)

    ellipsoid.plot_shape_mask(ds)


def test_plot_fp_reference_figure():
    synthetic_discrete_objects.create_figure(
        temp_files_path=TEMP_DATA_PATH, dx=10.0, l_shear_max=1000.0, length_max=4.0
    )
