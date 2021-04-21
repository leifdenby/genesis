from . import shapes as plot_shapes
from ..minkowski import analytical as minkowski_analytical
from ....utils.plot_types import multi_jointplot
from ....utils import xarray as xarray_utils

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from matplotlib.projections import get_projection_class
import seaborn as sns
import numpy as np
import xarray as xr
import warnings


def plot_reference(  # noqa
    ax,
    shape,
    lm_range=slice(1.0 / 4.0, 9),
    linestyle="-",
    marker="o",
    x_pos=0.85,
    y_pos=0.6,
    scale=0.4,
    lm_diagram=2.5,
    include_shape_diagram=True,
    calc_kwargs={},
    lm_label_sel=lambda lm: lm,  # usually outside since lm=1 is at the origin
    reference_data_path=None,
    legend_fn=None,
    **kwargs,
):
    """
    If `reference_data_path` is provided the calculations will be cache there
    """
    # the ellipsoid lines are plotted using the spheroid lambda values and so
    # we show the spheroid lines too, the `lambda_shape` is the one producing
    # the reference line where two axis are the same (e.g. either a spheroid or
    # a cylinder)
    plot_ellipsoid_lines = False
    if shape == "ellipsoid":
        plot_ellipsoid_lines = True
        lambda_shape = "spheroid"
    else:
        lambda_shape = shape

    try:
        fn = getattr(plot_shapes, shape)
    except AttributeError:
        raise NotImplementedError(shape)

    fig = ax.get_figure()

    if type(lm_range) == tuple and len(lm_range) == 2:
        lm_range = slice(*lm_range)

    reference_lines = {}

    ds = xarray_utils.cache_to_file(
        path=reference_data_path,
        fname=f"fp_scales_reference_{lambda_shape}__{lm_range.start}_{lm_range.stop}.nc",
        func=minkowski_analytical.calc_analytical_scales,
        shape=lambda_shape,
        **calc_kwargs,
    )

    if lm_range is not None:
        ds = ds.swap_dims(dict(i="lm")).sel(lm=lm_range).swap_dims(dict(lm="i"))

    reference_lines[lambda_shape] = ds

    F = ds.filamentarity
    P = ds.planarity

    shape_label = rf"{lambda_shape}: $c=\lambda a, b=a$"
    (line,) = ax.plot(P, F, linestyle=linestyle, label=shape_label, **kwargs)

    if "color" not in kwargs:
        kwargs["color"] = line.get_color()

    # calculate the lambda values to highlight
    def _find_fractions_of_two(v):
        return 2.0 ** np.array(list(set(np.log2(v).astype(int))))

    def _find_integer_values(v):
        return np.sort(np.array(list(set(v.astype(int)))))

    lm_ = _find_fractions_of_two(ds.lm.values)

    marker_points = []

    for lm_pt in lm_:
        ds_ = ds.swap_dims(dict(i="lm")).sel(lm=lm_pt, method="nearest")
        x_, y_ = ds_.planarity, ds_.filamentarity

        if shape != "cylinder":
            marker_points.append(
                [(ds_.planarity, ds_.filamentarity), (ds_.a, ds_.a, ds_.c)]
            )

        if marker != "shape":
            ax.plot(x_, y_, marker=marker, label="", **kwargs)
        if lm_pt > 1:
            s = "{:.0f}".format(lm_pt)
            dx, dy = -4, 0
            ha = "right"
        elif lm_pt == 1:
            s = "{:.0f}".format(lm_pt)
            dx, dy = -2, 10
            ha = "center"
        else:
            s = "1/{:.0f}".format(1.0 / lm_pt)
            dx, dy = 0, -14
            ha = "center"
        if lm_label_sel(lm_pt):
            s = r"$\lambda=$" + s
        ax.annotate(
            s,
            (x_, y_),
            color=kwargs["color"],
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
        )

    def _pt_transform(coord):
        return (ax.transData + fig.transFigure.inverted()).transform(coord)

    N_points_calc = calc_kwargs.get("N_points", 100)
    if plot_ellipsoid_lines:
        ellipsoids = []
        for lm_pt in lm_[lm_ > 1.0]:
            calc_kwargs["N_points"] = int(N_points_calc / 5 * int(lm_pt ** 0.4))
            kwargs["linestyle"] = ":"
            if len(ellipsoids) == 0:
                kwargs["label"] = (
                    "ellipsoid: $c = r_0 \\lambda^{2/3}, b = \\gamma a$, "
                    "$a\\leq b \\leq c$,"
                    "\n"
                    "               $b \\in [r_0/\\lambda^{1/3};c], a \\in [r_0/c^2; r_0/\\lambda^{1/3}]$"
                )
            else:
                kwargs.pop("label", None)

            ds_ellip = xarray_utils.cache_to_file(
                path=reference_data_path,
                fname=f"fp_scales_reference_ellipsoid_{lm_pt}.nc",
                func=minkowski_analytical.calc_analytical_scales,
                shape="ellipsoid",
                lm=lm_pt,
                **calc_kwargs,
            )
            ellipsoids.append(ds_ellip)

            ax.plot(ds_ellip.planarity, ds_ellip.filamentarity, zorder=0.9, **kwargs)

            gamma = _find_integer_values(ds_ellip.gamma.values)
            for n, gamma_ in enumerate(gamma):
                if not gamma_ > 1.0:
                    continue

                ds_ = ds_ellip.swap_dims(dict(i="gamma")).sel(
                    gamma=gamma_, method="nearest"
                )

                if marker == "shape":
                    if shape == "cylinder":
                        raise NotImplementedError(shape)
                    else:
                        x_, y_ = ds_.planarity, ds_.filamentarity
                        marker_points.append(
                            [(x_, y_), np.array([ds_.a, ds_.b, ds_.c])]
                        )

                # if lm_pt == lm_[lm_ >= 1.0][-1]:
                ax.annotate(
                    fr"$\gamma={int(gamma_)}$",
                    (ds_.planarity, ds_.filamentarity),
                    color=kwargs["color"],
                    zorder=1.0,
                    xytext=(0, ds_.c / 35),
                    textcoords="offset points",
                    horizontalalignment="center",
                )

                if not gamma_ > 1.0:
                    continue

                if marker != "shape":
                    ax.plot(
                        ds_.planarity,
                        ds_.filamentarity,
                        marker=".",
                        zorder=0.9,
                        **kwargs,
                    )

        ds_ellipsoids = xr.concat(ellipsoids, dim="lm")
        reference_lines["ellipsoid"] = ds_ellipsoids

    if include_shape_diagram:
        l = scale / 2.0  # noqa

        shape_kws = {}
        if shape == "ellipsoid":
            shape_kws["r2_label"] = "$a$"

        fn(
            ax,
            x_pos,
            y_pos,
            l=l,  # noqa
            r=l / lm_diagram,
            color=line.get_color(),
            h_label=r"$c$",
            r_label=r"$b$",
            **shape_kws,
        )

    if marker == "shape":
        for ((x_, y_), (a, b, c)) in marker_points:
            if x_ > ds.planarity.max() * 1.4 or y_ > ds.filamentarity.max():
                continue

            size = scale / 2.0
            xp, yh = _pt_transform((x_, y_))
            with warnings.catch_warnings():
                # this will suppress all warnings in this block
                warnings.simplefilter("ignore")
                ax_inset = inset_axes(
                    parent_axes=ax,
                    width="100%",
                    height="100%",
                    bbox_to_anchor=(x_ - size / 2, y_ - size / 2, size, size),
                    bbox_transform=ax.transData,
                    borderpad=0,
                    axes_class=get_projection_class("3d"),
                    axes_kwargs=dict(facecolor="none"),
                )
            radii = np.array([a, b, c])
            if shape == "ellipsoid":
                r_max = np.max(
                    [
                        ds_ellip.a.max(),
                        ds_ellip.b.max(),
                        ds_ellip.c.max(),
                    ]
                )
            else:
                r_max = np.max(
                    [
                        ds.a.max(),
                        ds.c.max(),
                    ]
                )
            radii *= 50.0 / r_max

            plot_shapes.spheroid_3d(
                ax=ax_inset,
                radii=radii,
                rotation=[0.0, 0.0],
                plot_axes=True,
                resolution=50,
                show_grid_axes=False,
                color=kwargs["color"],
            )

    xlabel = ax.get_xlabel()
    if xlabel:
        if xlabel not in ["planarity", "planarity [1]"]:
            raise Exception(xlabel)
    else:
        ax.set_xlabel("planarity")

    ylabel = ax.get_ylabel()
    if ylabel:
        if ylabel not in ["filamentarity", "filamentarity [1]"]:
            raise Exception(ylabel)
    else:
        ax.set_ylabel("filamentarity")

    legend_text = (
        "reference lines\n"
        "fixed volume: $v_0=\\frac{4}{3}\\pi a b c=\\frac{4}{3}\\pi r_0^3$, "
    )

    if legend_fn is None:
        legend_fn = ax.legend

    handles, labels = ax.get_legend_handles_labels()
    legend = legend_fn(
        title=legend_text,
        labels=labels,
        handles=handles,
        loc="upper left",
        bbox_to_anchor=[1.2, 1.0],
    )
    plt.setp(legend.get_title(), multialignment="center")

    return reference_lines


def fp_plot(ds, lm_range=None, reference_shape="spheroid"):
    g = sns.jointplot("planarity", "filamentarity", data=ds, stat_func=None, marker=".")

    g.plot_joint(sns.kdeplot, cmap="Blues", zorder=0)
    ax = g.ax_joint
    ax.text(
        0.95,
        0.95,
        "{} objects".format(len(ds.object_id)),
        transform=ax.transAxes,
        horizontalalignment="right",
        bbox=dict(facecolor="white", gamma=0.8, edgecolor="none"),
    )
    plot_reference(
        ax=ax,
        shape=reference_shape,
        lm_range=lm_range,
        color="red",
        linestyle="--",
        marker=".",
        x_pos=0.9,
    )
    ax.set_ylim(-0.01, ds.filamentarity.max())
    ax.set_xlim(-0.01, ds.planarity.max())
    return g


def main(ds, auto_scale=True, reference_shape="spheroid"):
    """
    Create a filamentarity-planarity joint plot using the `dataset` attribute
    of `ds` for the hue
    """
    x = "planarity"
    y = "filamentarity"

    assert x in ds
    assert y in ds

    if "dataset" not in ds:
        g = fp_plot(ds)
    else:

        g = multi_jointplot(
            x="planarity",
            y="filamentarity",
            z="dataset",
            ds=ds,
            joint_type="kde",
            joint_kwargs=dict(marker=".", alpha=0.2),
        )

        # store a reference to labels we already have in the legend
        jp_labels = [t.get_text() for t in g.ax_joint.get_legend().texts]
        jp_handles = g.ax_joint.get_legend().legendHandles

        def legend_fn(labels, handles, **kwargs):
            labels = list(labels) + jp_labels
            handles = list(handles) + jp_handles
            # kwargs['bbox_to_anchor']=[0.5, -0.2]
            # kwargs['loc']="lower center"
            # kwargs['ncol'] = 2
            return g.ax_joint.legend(
                labels=labels,
                handles=handles,
                **kwargs,
            )

        plot_reference(
            ax=g.ax_joint,
            shape=reference_shape,
            color="black",
            lm_label_sel=lambda lm: lm > 2.0,
            legend_fn=legend_fn,
        )

    if auto_scale:
        g.ax_joint.set_xlim(-0.0, 0.45)
        g.ax_joint.set_ylim(-0.0, 0.9)

    return g
