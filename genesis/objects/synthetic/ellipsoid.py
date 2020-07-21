import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def Rot_x(t):
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def Rot_z(t):
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def rotate_points(p, theta, phi):
    p_flat = np.array([np.asarray(c).flatten() for c in p])

    p_flat_rot = np.dot(Rot_z(-phi), np.dot(Rot_x(-theta), p_flat,))

    return [p_flat_rot[n].reshape(c.shape) for (n, c) in enumerate(p)]


def make_ellipsoid_mask(grid, a=2, b=4, c=1, theta=0.0, phi=0.0):
    x, y, z = [np.asarray(comp) for comp in grid]
    x_, y_, z_ = rotate_points(grid, theta=theta, phi=phi)

    return x_ ** 2.0 / a ** 2.0 + y_ ** 2.0 / b ** 2.0 + z_ ** 2.0 / c ** 2.0 < 1.0


def plot_shape(ds):
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


def find_ellipsoid_scales(da_mask):
    def I_func(x, y, z, m):
        return np.array([
            [np.sum(m * (y ** 2.0 + z ** 2.0)), np.sum(m * x * y), np.sum(m * x * z)],
            [np.sum(m * y * x), np.sum(m * (x ** 2.0 + z ** 2.0)), np.sum(m * y * z)],
            [np.sum(m * z * x), np.sum(m * z * y), np.sum(m * (y ** 2.0 + x ** 2.0))],
        ])

    if np.any(da_mask.isnull()):
        m = ~da_mask.isnull()
    else:
        m = da_mask

    # estimating volume assuming isotropic grid (!)
    mass = np.sum(np.count_nonzero(m))

    # need to center coordinates on center of mass
    if len(da_mask.x.shape) == 3:
        x_3d = da_mask.x
        y_3d = da_mask.y
        z_3d = da_mask.z
    else:
        x_3d, y_3d, z_3d = xr.broadcast(da_mask.x, da_mask.y, da_mask.z)

    x_c = x_3d.where(m, other=0.0).sum(dtype="float64") / mass
    y_c = y_3d.where(m, other=0.0).sum(dtype="float64") / mass
    z_c = z_3d.where(m, other=0.0).sum(dtype="float64") / mass

    x_ = x_3d - x_c
    y_ = y_3d - y_c
    z_ = z_3d - z_c

    I = I_func(x_, y_, z_, m)  # noqa

    la, v = np.linalg.eig(I)

    # sort eigenvectors by eigenvalue, the smallest eigenvalue will be the
    # principle axis (the eigenvalue represents the moment of inertia and mass
    # is most compact perpendicular to this direction)
    sort_idx = np.argsort(la)
    la = la[sort_idx]
    v = v[sort_idx]

    # estimate ellipsoid scales from moment of inertia (eigenvalues)
    a = np.sqrt((la[1] + la[2] - la[0]) * 5.0 / 2.0 * 1.0 / mass)
    b = np.sqrt((la[0] + la[2] - la[1]) * 5.0 / 2.0 * 1.0 / mass)
    c = np.sqrt((la[0] + la[1] - la[2]) * 5.0 / 2.0 * 1.0 / mass)

    def calc_angles(v):
        phi = np.arctan2(np.abs(v[1]), np.abs(v[0]))
        l_xy = np.sqrt(v[0] ** 2.0 + v[1] ** 2.0)
        theta = np.arctan2(np.abs(v[2]), l_xy)

        return phi, theta

    return la, v, (a, b, c), np.array([np.rad2deg(calc_angles(v[n])) for n in range(3)])


def _make_test_grid():
    lx, ly, lz = 100, 100, 100

    x_ = np.arange(-lx / 2, lx / 2, 1)
    y_ = np.arange(-ly / 2, ly / 2, 1)
    z_ = np.arange(-lz / 2, lz / 2, 1)

    ds = xr.Dataset(coords=dict(x=x_, y=y_, z=z_))

    ds["x_3d"], ds["y_3d"], ds["z_3d"] = xr.broadcast(ds.x, ds.y, ds.z)
    ds.atrs["lx"] = lx
    ds.atrs["ly"] = ly
    ds.atrs["lz"] = lz

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
    mask = make_ellipsoid_mask(
        grid=(ds.x_3d, ds.y_3d, ds.z_3d), theta=0, phi=0, a=a, b=b, c=c
    )

    ds["mask_rot"] = (ds.x_3d.dims, mask)

    plot_shape(ds)

    mask = make_ellipsoid_mask(
        grid=(ds.x_3d, ds.y_3d, ds.z_3d), theta=45.0 / 180 * 3.14, phi=0, a=a, b=b, c=c
    )
    ds["mask_rot"] = (ds.x_3d.dims, mask)

    plot_shape(ds)

    mask = make_ellipsoid_mask(
        grid=(ds.x_3d, ds.y_3d, ds.z_3d), theta=0, phi=60.0 / 180 * 3.14, a=a, b=b, c=c
    )
    ds["mask_rot"] = (ds.x_3d.dims, mask)

    plot_shape(ds)
