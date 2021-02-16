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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from ..minkowski import discrete as minkowski_discrete
from ...synthetic.discrete import make_mask
from .filamentarity_planarity import plot_reference as plot_fp_ref


TEMP_FILENAME_FORMAT = "{h}_{length}_{dx}_{l_shear}_{shape}.{filetype}"


def _add_text_fp_diagram(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("planarity")
    ax.set_ylabel("filamentarity")

    ax.text(0.25, 0.25, "$L \\approx W \\approx T$", ha="center")
    ax.text(0.25, 0.75, "$L > W \\approx T$", ha="center")
    ax.text(0.75, 0.75, "$L > W > T$", ha="center")
    ax.text(0.75, 0.25, "$L \\approx W > T$", ha="center")
    ax.set_xticks([])
    ax.set_yticks([])

    sns.despine(ax=ax)


def _make_filename(ds_params, filetype):
    h, length, dx, shape = (
        ds_params.h.values,
        ds_params.length.values,
        ds_params.dx.values,
        ds_params.shape.values,
    )
    l_shear = ds_params.l_shear.values
    return TEMP_FILENAME_FORMAT.format(
        h=h, length=length, dx=dx, l_shear=l_shear, shape=shape, filetype=filetype
    )


def _calc_scales(ds_params, output_path):
    """
    `ds_synth`: xr.Dataset which contains `mask`
    """
    fn = _make_filename(ds_params=ds_params, filetype="scales.nc")
    p = Path(output_path) / fn

    if not p.exists():
        ds_synth = _get_mask(ds_params, output_path=output_path)
        object_labels = ds_synth.mask.values.astype(int)
        ds_scales = minkowski_discrete.calc_scales(
            object_labels=object_labels, dx=ds_synth.dx
        )
        ds_scales.to_netcdf(str(p))
    else:
        ds_scales = xr.open_dataset(str(p))
    return ds_scales


def _create_single_mask_plot(ds_synth):
    fig, ax = plt.subplots()
    m = ds_synth.sel(y=0, method="nearest").mask
    m.where(m, other=np.nan).plot(y="z", ax=ax, add_colorbar=False, cmap="Greys_r")
    plt.gca().set_aspect(1)
    plt.axis("off")
    plt.title("")


def _get_mask_image(ds_params, output_path):
    fn = _make_filename(ds_params=ds_params, filetype="png")
    p = Path(output_path) / fn

    if not p.exists():
        ds_synth = _get_mask(ds_params, output_path=output_path)
        _create_single_mask_plot(ds_synth=ds_synth)
        plt.savefig(str(p))

    img = plt.imread(str(p))
    return img


def _get_mask(ds_params, output_path):
    fn = _make_filename(ds_params=ds_params, filetype="mask.nc")
    p = Path(output_path) / fn

    if not p.exists():
        h, length, dx, shape = (
            ds_params.h.values,
            ds_params.length.values,
            ds_params.dx.values,
            ds_params.shape.values,
        )
        l_shear = ds_params.l_shear.values
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
    xlim=(-0.01, 0.25),
    ylim=(-0.01, 0.55),
    aspect=0.5,
    figsize=(5, 5),
):
    if reference_shape == "ellipsoid":
        x_pos_shape = 1.2
        y_pos_shape = 0.3
    else:
        x_pos_shape = 0.8
        y_pos_shape = 0.5

    temp_files_path = Path(temp_files_path).expanduser()

    temp_files_path.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect(1)
    reference_lines = plot_fp_ref(
        ax=ax,
        shape=reference_shape,
        lm_range=slice(1.0 / 4.0, 9),
        calc_kwargs=dict(N_points=400),
        reference_data_path=temp_files_path,
        include_shape_diagram="at_reference_points",
        marker="shape",
        x_pos=x_pos_shape,
        y_pos=y_pos_shape,
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

    datasets = []
    for i in iterator(range(len(ds_flat.i))):
        ds_params = ds_flat.isel(i=i)

        img = _get_mask_image(ds_params=ds_params, output_path=temp_files_path)
        ds_scales = _calc_scales(ds_params=ds_params, output_path=temp_files_path)
        ds_scales = ds_scales.assign_coords(object_id=[i])

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
        datasets.append(xr.merge([ds_scales, ds_params]))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(aspect)
    sns.despine()

    ax_inset = inset_axes(
        parent_axes=ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.47, ylim[0], 0.16, 0.25),
        bbox_transform=ax.transData,
        borderpad=0,
        axes_kwargs=dict(facecolor="none"),
    )
    _add_text_fp_diagram(ax=ax_inset)

    return ax, xr.concat(datasets, dim="object_id"), reference_lines


if __name__ == "__main__":
    fig = create_figure(show_progress=True)
    output_fn = "fp_synthetic_examples.png"
    fig.savefig(output_fn, dpi=400)
    print("Wrote {}".format(output_fn))
