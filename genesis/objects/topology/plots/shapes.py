"""
Routines for plotting annotated outlines of 3D shapes with prescribed
characteristic scales
"""
from math import cos, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Ellipse


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
    r2_label=":r",
    h_label="h",
    render_back=True,
):
    """
    Spheroid rendered at (x_c, y_c) with height `2*l` and width `2*r`. The length
    of the projected cutout y-axis is given by `y_axis_3d_len`, changing this
    value changes the effective viewing angle.
    """
    if r2_label == ":r":
        r2_label = r_label

    kwargs = dict(
        transform=ax.transAxes,
        facecolor="None",
        edgecolor=color,
        clip_on=False,
    )

    ln_kwargs = dict(
        transform=ax.transAxes,
        color=color,
        clip_on=False,
    )

    w = 3 * r

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
        clip_on=False,
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
    ax.add_line(plt.Line2D((x_c, x_c + w / 2), (y_c, y_c), ls="--", **ln_kwargs))
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
        r2_label,
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


def ellipsoid(*args, **kwargs):
    spheroid(*args, **kwargs)


def _make_3d_rotation_matrix(theta, phi):
    """
    theta: y-axis rotation, beta, "pitch"
    phi: z-axis rotation, alpha, "yaw"

    ref: https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    """
    alpha = phi
    beta = -theta

    Rz = np.array(
        [[cos(alpha), -sin(alpha), 0], [sin(alpha), cos(alpha), 0], [0, 0, 1]]
    )

    Ry = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])

    R = np.dot(Ry, Rz)
    return R


def spheroid_3d(
    radii,
    rotation,
    ax=None,
    plot_axes=False,
    color="b",
    alpha=0.2,
    show_grid_axes=True,
    resolution=100,
):
    """
    Plot an ellipsoid. Rotation is expected to be `[theta, phi]` where `theta`
    is the angle from the z-axis and `phi` is the angle of rotation around the
    z-axis

    based of https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py
    """
    R_rot = _make_3d_rotation_matrix(*rotation)
    center = [50.0, 50.0, 50.0]

    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

    N = 100

    u = np.linspace(0.0, 2.0 * np.pi, N)
    v = np.linspace(0.0, np.pi, N)

    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = (
                np.dot([x[i, j], y[i, j], z[i, j]], R_rot) + center
            )

    if plot_axes:
        # make some purdy axes
        axes = np.array(
            [[radii[0], 0.0, 0.0], [0.0, radii[1], 0.0], [0.0, 0.0, radii[2]]]
        )
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], R_rot)

        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=color)

    # plot ellipsoid
    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=4 * max(int(N / resolution), 1),
        cstride=4 * max(int(N / resolution), 1),
        color=color,
        alpha=alpha,
    )

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)

    if not show_grid_axes:
        ax.axis("off")

    return ax
