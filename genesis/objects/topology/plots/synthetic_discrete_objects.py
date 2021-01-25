# coding: utf-8
"""
Routines for plotting shape cross-sections of parametetrised synthetic 3D
shapes on a filamentarity-planarity plot of their properties
"""
if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

from ..minkowski import discrete as minkowski_discrete
from ...synthetic.discrete import make_mask
from .filamentarity_planarity import plot_reference as plot_fp_ref


IMAGE_FILENAME_FORMAT = "{h}_{length}_{dx}_{l_shear}_{shape}.{filetype}"


def _calc_scales(ds_params, output_path):
    """
    `ds_synth`: xr.Dataset which contains `mask`
    """
    ds_synth = _get_mask(ds_params, output_path=output_path)
    object_labels = ds_synth.mask.values.astype(int)
    scales = minkowski_discrete.calc_scales(object_labels=object_labels, dx=ds_synth.dx)
    return scales


def _create_single_mask_plot(ds_synth):
    fig, ax = plt.subplots()
    m = ds_synth.sel(y=0, method="nearest").mask
    m.where(m, other=np.nan).plot(y="z", ax=ax, add_colorbar=False, cmap="Greys_r")
    plt.gca().set_aspect(1)
    plt.axis("off")
    plt.title("")


def _get_mask_image(ds_params, output_path):
    h, length, dx, shape = (
        ds_params.h.values,
        ds_params.length.values,
        ds_params.dx.values,
        ds_params.shape.values,
    )
    l_shear = ds_params.l_shear.values
    fn_img = IMAGE_FILENAME_FORMAT.format(
        h=h, length=length, dx=dx, l_shear=l_shear, shape=shape, filetype="png"
    )
    p = Path(output_path) / fn_img

    if not p.exists():
        ds_synth = _get_mask(ds_params, output_path=output_path)
        _create_single_mask_plot(ds_synth=ds_synth)
        plt.savefig(str(p))

    img = plt.imread(str(p))
    return img


def _get_mask(ds_params, output_path):
    h, length, dx, shape = (
        ds_params.h.values,
        ds_params.length.values,
        ds_params.dx.values,
        ds_params.shape.values,
    )
    l_shear = ds_params.l_shear.values
    fn_nc = IMAGE_FILENAME_FORMAT.format(
        h=h, length=length, dx=dx, l_shear=l_shear, shape=shape, filetype="nc"
    )

    p = Path(output_path) / fn_nc
    if not p.exists():
        ds_synth = make_mask(h=h, length=length, dx=dx, shape=shape, l_shear=l_shear)
        ds_synth.to_netcdf(str(p))
    else:
        ds_synth = xr.open_dataset(str(p))

    return ds_synth


def create_figure(
    reference_shape="ellipsoid",
    temp_files_path=".",
    show_progress=False,
    length_max=8.0,
    l_shear_max=1500.0,
    dx=4.0,
):
    Path(temp_files_path).mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect(1)
    plot_fp_ref(
        ax=ax,
        shape=reference_shape,
        lm_range=slice(1.0 / 4.0, 9),
        calc_kwargs=dict(N_points=400),
    )

    ds_study = xr.Dataset(
        coords=dict(
            h=[1000.0],
            length=np.arange(2.0, length_max, 1.0),
            l_shear=np.arange(0.0, l_shear_max, 500.0),
            dx=[
                dx,
            ],
            shape=["thermal"],
        )
    )

    def format_length(v):
        if v == np.inf:
            return r"$\infty$"
        else:
            return r"{}m".format(v)

    # create a new flattened index
    ds_flat = ds_study.stack(i=ds_study.dims).reset_index("i")

    if show_progress:
        iterator = tqdm.tqdm
    else:
        iterator = lambda v: v  # noqa

    for i in iterator(range(len(ds_flat.i))):
        ds_params = ds_flat.isel(i=i)

        img = _get_mask_image(ds_params=ds_params, output_path=temp_files_path)
        ds_scales = _calc_scales(ds_params=ds_params, output_path=temp_files_path)

        lx, ly = 0.04, 0.04
        extent = np.array(
            [
                ds_scales.planarity - lx / 2.0,
                ds_scales.planarity + lx / 2.0,
                ds_scales.filamentarity - ly / 2.0,
                ds_scales.filamentarity + ly / 2.0,
            ]
        ).T[0]
        ax.imshow(img, extent=extent)

    ax.set_xlim(-0.01, 0.25)
    ax.set_ylim(-0.01, 0.55)
    ax.set_aspect(0.5)
    sns.despine()
    ax.legend(loc="upper right")

    return fig


if __name__ == "__main__":
    fig = create_figure(show_progress=True)
    output_fn = "fp_synthetic_examples.png"
    fig.savefig(output_fn, dpi=400)
    print("Wrote {}".format(output_fn))
