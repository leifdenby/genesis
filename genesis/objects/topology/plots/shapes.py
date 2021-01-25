"""
Routines for plotting annotated outlines of 3D shapes with prescribed
characteristic scales
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc

from math import sqrt


def cylinder(ax, x_c, y_c, l, r, color, r_label="r", h_label="h"):
    e_l = 0.1 * l
    # sides
    ax.add_line(
        plt.Line2D((x_c - r, x_c - r), (y_c - l / 2.0, y_c + l / 2.0), color=color)
    )
    ax.add_line(
        plt.Line2D((x_c + r, x_c + r), (y_c - l / 2.0, y_c + l / 2.0), color=color)
    )
    # centerline
    ax.add_line(
        plt.Line2D((x_c, x_c), (y_c - l / 2.0, y_c + l / 2.0), color=color, ls="--")
    )

    # radius indicator
    ax.add_line(plt.Line2D((x_c - r, x_c), (y_c, y_c), color=color, ls="--"))

    # ends
    c = ax.get_lines()[-1].get_color()
    ax.add_patch(
        Ellipse(
            (x_c, y_c - l / 2.0),
            r * 2,
            e_l,
            facecolor="None",
            edgecolor=c,
            linewidth=2,
            linestyle=":",
            alpha=0.4,
        )
    )

    ax.add_patch(
        Arc(
            (x_c, y_c - l / 2.0),
            r * 2,
            e_l,
            facecolor="None",
            edgecolor=c,
            linewidth=2,
            theta1=180,
            theta2=360,
        )
    )

    ax.add_patch(
        Ellipse(
            (x_c, y_c + l / 2.0), r * 2, e_l, facecolor="None", edgecolor=c, linewidth=2
        )
    )

    # labels
    ax.annotate(
        r_label,
        (x_c - r / 2.0, y_c),
        color=c,
        xytext=(0, 6),
        textcoords="offset points",
    )

    ax.annotate(
        h_label, (x_c, y_c + l / 4), color=c, xytext=(6, 0), textcoords="offset points"
    )


def spheroid(
    ax,
    x_c,
    y_c,
    l,
    r,
    color,
    y_axis_3d_len=0.05,
    r_label="r",
    h_label="h",
    render_back=True,
):
    """
    Spheroid rendered at (x_c, y_c) with height `2*l` and width `2*r`. The length
    of the projected cutout y-axis is given by `y_axis_3d_len`, changing this
    value changes the effective viewing angle.
    """

    kwargs = dict(
        transform=ax.transAxes,
        facecolor="None",
        edgecolor=color,
    )

    ln_kwargs = dict(
        transform=ax.transAxes,
        color=color,
    )

    w = 2 * r

    # arcs connecting to y-axis (into page) and widths associated
    # y-axis length projection into xz plane
    c = y_axis_3d_len
    l_yz_axis = sqrt(c ** 2.0 / 2.0)
    # ellipsoid minor axis in projected x-plane
    l_yz_arc = sqrt(l ** 2.0 * c ** 2.0 / (2 * l ** 2.0 - c ** 2.0))
    # ellipsoid minor axis in projected y-plane
    l_xy_arc = sqrt(r ** 2.0 * c ** 2.0 / (2 * r ** 2.0 - c ** 2.0))

    # add background white oval incase there are some lines behind
    b_pad = 0.05
    bckgrnd_patch = Ellipse(
        (x_c, y_c),
        w + b_pad,
        l * 2 + b_pad,
        facecolor="white",
        transform=ax.transAxes,
        alpha=0.9,
    )
    ax.add_patch(bckgrnd_patch)

    # yz-plane arc
    if render_back:
        ax.add_patch(
            Arc(
                (x_c, y_c),
                l_yz_arc * 2,
                l * 2,
                linewidth=2,
                linestyle=":",
                alpha=0.4,
                **kwargs
            )
        )

    ax.add_patch(
        Arc(
            (x_c, y_c),
            l_yz_arc * 2,
            l * 2,
            linewidth=2,
            linestyle="-",
            theta1=90.0,
            theta2=270.0,
            **kwargs
        )
    )

    # xy-plane arc
    if render_back:
        ax.add_patch(
            Arc(
                (x_c, y_c),
                w,
                l_xy_arc * 2,
                linewidth=2,
                linestyle=":",
                theta1=0,
                theta2=180,
                alpha=0.4,
                **kwargs
            )
        )

    a_xy = Arc(
        (x_c, y_c),
        w,
        l_xy_arc * 2,
        linewidth=2,
        linestyle="-",
        theta1=180,
        theta2=360,
        **kwargs
    )
    ax.add_patch(a_xy)

    # xz-plane edge
    ax.add_patch(Ellipse((x_c, y_c), w, l * 2, linewidth=2, **kwargs))

    # line along x-axis
    ax.add_line(plt.Line2D((x_c, x_c + r), (y_c, y_c), ls="--", **ln_kwargs))
    # line along z-axis
    ax.add_line(plt.Line2D((x_c, x_c), (y_c, y_c + l), ls="--", **ln_kwargs))
    # line along y-axis (into page)
    ax.add_line(
        plt.Line2D((x_c, x_c - l_yz_axis), (y_c, y_c - l_yz_axis), ls="--", **ln_kwargs)
    )

    # labels
    ax.annotate(
        r_label,
        (x_c + r / 2.0, y_c),
        xytext=(0, 6),
        textcoords="offset points",
        xycoords="axes fraction",
        ha="center",
        **ln_kwargs
    )

    ax.annotate(
        r_label,
        (x_c - l_yz_arc * 0.5, y_c - 0.25 * l_yz_axis),
        xytext=(0, 10),
        textcoords="offset points",
        xycoords="axes fraction",
        ha="center",
        **ln_kwargs
    )

    ax.annotate(
        h_label,
        (x_c, y_c + l / 2),
        xytext=(4, 0),
        textcoords="offset points",
        xycoords="axes fraction",
        **ln_kwargs
    )
