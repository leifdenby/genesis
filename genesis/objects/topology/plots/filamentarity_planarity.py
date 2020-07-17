from genesis.objects.topology.plots import shapes as plot_shapes
from genesis.objects.topology import minkowski_analytical
from genesis.utils.plot_types import multi_jointplot

import seaborn as sns
import numpy as np


def plot_reference(
    ax,
    shape,
    lm_range=None,
    linestyle="-",
    marker="o",
    x_pos=0.85,
    y_pos=0.6,
    scale=0.4,
    lm_diagram=2.5,
    include_shape_diagram=True,
    calc_kwargs={},
    **kwargs
):
    # the ellipsoid lines are plotted using the spheroid lambda values and so
    # we show the spheroid lines too
    plot_ellipsoid_lines = False
    if shape == 'ellipsoid':
        shape = 'spheroid'
        plot_ellipsoid_lines = True

    try:
        fn = getattr(plot_shapes, shape)
    except AttributeError:
        raise NotImplementedError(shape)

    ds = minkowski_analytical.calc_analytical_scales(shape=shape, **calc_kwargs)
    if lm_range is not None:
        ds = ds.swap_dims(dict(i="lm")).sel(lm=lm_range).swap_dims(dict(lm="i"))

    F = ds.filamentarity
    P = ds.planarity

    (line,) = ax.plot(P, F, linestyle=linestyle, label=shape, **kwargs)

    if "color" not in kwargs:
        kwargs["color"] = line.get_color()

    # calculate the lambda values to highlight
    def _find_fractions_of_two(v):
        return 2.**np.array(list(set(np.log2(v).astype(int))))

    def _find_integer_values(v):
        return np.sort(np.array(list(set(v.astype(int)))))

    lm_ = _find_fractions_of_two(ds.lm.values)

    for lm_pt in lm_:
        ds_ = ds.swap_dims(dict(i="lm")).sel(lm=lm_pt, method='nearest')
        x_, y_ = ds_.planarity, ds_.filamentarity

        lm_max = int(ds.lm.values.max())

        ax.plot(x_, y_, marker=marker, label="", **kwargs)
        if lm_pt >= 1:
            s = "{:.0f}".format(lm_pt)
            dx, dy = -4, 0
            ha = "right"
        else:
            s = "1/{:.0f}".format(1.0 / lm_pt)
            dx, dy = 0, -14
            ha = "center"
        if lm_pt == lm_max:
            s = r"$\lambda=$" + s
        ax.annotate(
            s,
            (x_, y_),
            color=kwargs["color"],
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
        )

    N_points_calc = calc_kwargs.get('N_points', 100)
    if plot_ellipsoid_lines:
        for lm_pt in lm_[lm_ > 1.0]:
            calc_kwargs['N_points'] = int(N_points_calc/5 * int(lm_pt**0.4))
            kwargs['linestyle'] = ":"
            ds_ellip = minkowski_analytical.calc_analytical_scales(shape="ellipsoid", lm=lm_pt, **calc_kwargs)
            ax.plot(ds_ellip.planarity, ds_ellip.filamentarity, zorder=0.9, **kwargs)

            alpha = _find_integer_values(ds_ellip.alpha.values)
            for n, a_ in enumerate(alpha[alpha > 1.0]):
                ds_ = ds_ellip.swap_dims(dict(i="alpha")).sel(alpha=a_, method='nearest')
                ax.plot(ds_.planarity, ds_.filamentarity, marker='.', zorder=0.9, **kwargs)

                if n > 3:
                    break
                if lm_pt == lm_[lm_ > 1.0][-1]:
                    ax.annotate(
                        fr"$\alpha={int(a_)}$",
                        (ds_.planarity, ds_.filamentarity),
                        color=kwargs["color"], zorder=1.0,
                        xytext=(0, 5),
                        textcoords="offset points",
                        horizontalalignment="center",
                    )

    if include_shape_diagram:
        l = scale / 2.0  # noqa

        fn(
            ax,
            x_pos,
            y_pos,
            l=l,  # noqa
            r=l / lm_diagram,
            color=line.get_color(),
            h_label=r"$\lambda r$",
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


def fp_plot(ds, lm_range=None):
    g = sns.jointplot("planarity", "filamentarity", data=ds, stat_func=None, marker=".")

    g.plot_joint(sns.kdeplot, cmap="Blues", zorder=0)
    ax = g.ax_joint
    ax.text(
        0.95,
        0.95,
        "{} objects".format(len(ds.object_id)),
        transform=ax.transAxes,
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    plot_reference(
        ax=ax,
        shape="spheroid",
        lm_range=lm_range,
        color="red",
        linestyle="--",
        marker=".",
        x_pos=0.9,
    )
    ax.set_ylim(-0.01, ds.filamentarity.max())
    ax.set_xlim(-0.01, ds.planarity.max())
    return g


def main(ds, auto_scale=True):
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
            x="planarity", y="filamentarity", z="dataset", ds=ds, joint_type="kde"
        )

        LABEL_FORMAT = "{name}: {count} objects"
        g.ax_joint.legend(
            labels=[
                LABEL_FORMAT.format(
                    name=d.item(),
                    count=int(
                        ds.sel(dataset=d).dropna(dim="object_id").object_id.count()
                    ),
                )
                for d in ds.dataset
            ],
            bbox_to_anchor=[0.5, -0.2],
            loc="lower center",
            ncol=2,
        )

        plot_reference(ax=g.ax_joint, shape="spheroid", color="black")

    if auto_scale:
        g.ax_joint.set_xlim(-0.0, 0.45)
        g.ax_joint.set_ylim(-0.0, 0.9)
