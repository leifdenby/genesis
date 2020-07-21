# -*- coding: utf-8 -*-
import os
import xarray as xr
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

from ..utils import find_grid_spacing


def render_mask_as_3d_voxels(da_mask, ax=None, center_xy_pos=False, alpha=0.5):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection="3d")
    else:
        if not hasattr(ax, "voxels"):
            raise Exception("The provided axes must have `projection='3d'` set")

    dx = dy = dz = find_grid_spacing(da_mask)

    dims = ["x", "y", "z"]
    m_obj = da_mask.transpose(*dims).values

    x_c = da_mask[dims[0]]
    y_c = da_mask[dims[1]]
    z_c = da_mask[dims[2]]

    if center_xy_pos:
        if type(center_xy_pos) == bool:
            x_c -= x_c.mean()
            y_c -= y_c.mean()
        elif len(center_xy_pos) == 2:
            x_c -= center_xy_pos[0]
            y_c -= center_xy_pos[1]
        else:
            raise NotImplementedError(center_xy_pos)

    x = np.empty(np.array(x_c.shape) + 1)
    y = np.empty(np.array(y_c.shape) + 1)
    z = np.empty(np.array(z_c.shape) + 1)

    x[:-1] = x_c - 0.5 * dx
    y[:-1] = y_c - 0.5 * dy
    z[:-1] = z_c - 0.5 * dz
    x[-1] = x_c[-1] + 0.5 * dx
    y[-1] = y_c[-1] + 0.5 * dy
    z[-1] = z_c[-1] + 0.5 * dz

    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    m_edge = (
        (m_obj != np.roll(m_obj, axis=0, shift=1))
        | (m_obj != np.roll(m_obj, axis=0, shift=-1))
        | (m_obj != np.roll(m_obj, axis=1, shift=1))
        | (m_obj != np.roll(m_obj, axis=1, shift=-1))
        | (m_obj != np.roll(m_obj, axis=2, shift=1))
        | (m_obj != np.roll(m_obj, axis=2, shift=-1))
    )

    colors = np.zeros(list(m_obj.shape) + [4,], dtype=np.float32)
    colors[m_edge, 0] = 0
    colors[m_edge, 1] = 1
    colors[m_edge, 2] = 0
    colors[m_edge, 3] = alpha

    _ = ax.voxels(x, y, z, m_obj, facecolors=colors, edgecolors=[0, 0, 0, 0.5 * alpha])
    ax.set_xlabel(xr.plot.utils.label_from_attrs(x_c))
    ax.set_ylabel(xr.plot.utils.label_from_attrs(y_c))
    ax.set_zlabel(xr.plot.utils.label_from_attrs(z_c))

    return ax


def plot_orientation_triangle(ax, x_cl, y_cl, z_cl, phi, theta):
    x0 = float(x_cl[0])
    y0 = float(y_cl[0])
    z0 = float(z_cl[0])

    phi = float(phi)
    theta = float(theta)

    x_or = x0 + (z_cl - z0) * np.tan(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
    y_or = y0 + (z_cl - z0) * np.tan(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))

    ax.plot(x_or, y_or, z_cl, color="blue", linestyle="--")
    ax.plot(x_or, y_or, np.zeros_like(z_cl), color="blue", linestyle=":")
    ax.plot(
        x_or[-1].values * np.ones_like(x_or),
        y_or[-1].values * np.ones_like(y_or),
        z_cl,
        color="blue",
        linestyle=":",
    )

    # surface arc
    xl = x_cl.max() - x_cl.min()
    yl = y_cl.max() - y_cl.min()
    l_max = np.sqrt(xl ** 2.0 + yl ** 2.0)
    r_srf_arc = 0.2 * float(l_max)

    t_ = np.linspace(0.0, np.deg2rad(phi), 100)

    # surface angle plot for phi
    x_srf_arc = x0 + r_srf_arc * np.cos(t_)
    y_src_arc = y0 + r_srf_arc * np.sin(t_)
    ax.plot(x_srf_arc, y_src_arc, np.zeros_like(x_srf_arc), color="blue", linestyle="-")
    ax.plot(
        x0 + np.linspace(0, 1.5 * r_srf_arc, 100),
        y0 + np.zeros_like(y_src_arc),
        np.zeros_like(x_srf_arc),
        color="blue",
        linestyle="-",
    )

    ax.text(
        x0 + 1.5 * r_srf_arc * np.cos(t_.mean()),
        y0 + 1.5 * r_srf_arc * np.sin(t_.mean()),
        0,
        r"$\phi$",
        color="blue",
    )

    # elevated arc plot for theta

    t_ = np.linspace(np.deg2rad(90 - theta), np.deg2rad(90), 100)
    print(phi)

    x_arc = x0 + r_srf_arc * np.cos(np.deg2rad(phi)) * np.cos(t_)
    y_arc = y0 + r_srf_arc * np.sin(np.deg2rad(phi)) * np.cos(t_)
    z_arc = z0 + r_srf_arc * np.sin(t_)

    ax.plot(
        x_arc, y_arc, z_arc, color="blue", linestyle="-",
    )
    ax.plot([x0, x0], [y0, y0], [z0, z0 + r_srf_arc * 2.0], color="blue", linestyle="-")

    ax.text(
        x0 + 0.5 * r_srf_arc * np.cos(np.deg2rad(phi)),
        y0 + 0.5 * r_srf_arc * np.sin(np.deg2rad(phi)),
        z0 + 1.5 * r_srf_arc,
        r"$\theta$",
        color="blue",
        horizontalalignment="center",
    )


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument("object_file", type=str)
    argparser.add_argument("--object-id", type=int, required=True)
    argparser.add_argument("--center-xy-pos", action="store_true")

    args = argparser.parse_args()

    object_file = args.object_file.replace(".nc", "")

    if not "objects" in object_file:
        raise Exception()

    base_name, objects_mask = object_file.split(".objects.")

    fn_objects = "{}.nc".format(object_file)
    if not os.path.exists(fn_objects):
        raise Exception("Couldn't find objects file `{}`".format(fn_objects))
    da_objects = xr.open_dataarray(fn_objects, decode_times=False)

    if not args.object_id in da_objects.values:
        raise Exception()

    obj_id = args.object_id
    da_obj = da_objects.where(da_objects == obj_id, drop=True)
    da_mask = da_obj == obj_id

    da_mask = da_mask.rename(dict(xt="x", yt="y", zt="z"))

    ax = render_mask_as_3d_voxels(da_mask=da_mask, center_xy_pos=args.center_xy_pos)

    plt.show()
