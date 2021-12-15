import matplotlib.pyplot as plt
import numpy as np


def Rot_x(t):
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def Rot_z(t):
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def rotate_points(p, theta, phi):
    p_flat = np.array([np.asarray(c).flatten() for c in p])

    p_flat_rot = np.dot(
        Rot_z(-phi),
        np.dot(
            Rot_x(-theta),
            p_flat,
        ),
    )

    return [p_flat_rot[n].reshape(c.shape) for (n, c) in enumerate(p)]


def make_mask(grid, a=2, b=4, c=1, theta=0.0, phi=0.0):
    x, y, z = [np.asarray(comp) for comp in grid]
    x_, y_, z_ = rotate_points(grid, theta=theta, phi=phi)

    return x_ ** 2.0 / a ** 2.0 + y_ ** 2.0 / b ** 2.0 + z_ ** 2.0 / c ** 2.0 < 1.0


def plot_shape_mask(ds):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(7, 5))

    for n, d in enumerate(ds.dims):
        if "z" not in d:
            kwargs = dict(y="z")
        else:
            kwargs = dict()
        ax = axes[n // 2][n % 2]
        ds_ = ds.sel(**{d: 0, "method": "nearest"}).mask_rot
        ds_.plot(ax=ax, **kwargs)
        ax.set_aspect(1)
    fig.tight_layout()

    axes[1][1].set_aspect(1)
