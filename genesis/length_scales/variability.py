"""
Methods for quantifying coherent length-scales in the convection boundary-layer
"""
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from genesis import utils

model_name = "uclales"
case_name = "rico"


def _patch_average_splitting(d, s, shuffle_mask=False):
    """
    Split data d into 2**s patches and calculate mean and standard deviation
    over all patches
    """
    Nx, Ny = d.shape
    N = Nx
    assert Nx == Ny

    i, j = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")

    def make_mask(s):
        m = 2**s
        patch_size = int(N / m)
        mask = (i / (patch_size)) + m * (j / (patch_size))
        if np.any(np.isnan(mask)):
            raise Exception("Can't make mask of with this splitting")
        return mask, np.unique(mask), patch_size

    def apply_mask_shuffle(m):
        m_flat = m.reshape(Nx * Ny)
        np.random.shuffle(m_flat)
        return m_flat.reshape(Nx, Ny)

    mask, mask_entries, patch_size = make_mask(s=s)
    if shuffle_mask:
        mask = apply_mask_shuffle(m=mask)
    d_mean = ndimage.mean(d, labels=mask, index=mask_entries)
    d_std_err = ndimage.standard_deviation(d, labels=mask, index=mask_entries)

    return d_mean, d_std_err, mask_entries, patch_size


def _patch_average(d, patch_size, shuffle_mask=False):
    """
    Split data d into patches of size patch_size**2.0 (using a sliding window
    and discarding patches when window is outside of domain) and calculate mean
    and standard deviation over all patches
    """
    Nx, Ny = d.shape
    N = Nx
    assert Nx == Ny

    i, j = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")

    def make_mask(offset=0):
        """
        Make a mask with regions labelled in consecutive patches, remove any
        patches which because of offset window are outside of domain
        """
        assert offset < patch_size
        m = (N - offset) / patch_size
        mask = ((i - offset) / (patch_size)) + m * ((j - offset) / (patch_size))
        if np.any(np.isnan(mask)):
            raise Exception("Can't make mask of with this splitting")

        mask[m * patch_size + offset :, :] = -1
        mask[:, m * patch_size + offset :] = -1
        mask[:offset, :] = -1
        mask[:, :offset] = -1

        return mask, np.unique(mask)

    def apply_mask_shuffle(m):
        m_flat = m.reshape(Nx * Ny)
        np.random.shuffle(m_flat)
        return m_flat.reshape(Nx, Ny)

    _d_std_err = []

    for offset in range(0, patch_size, patch_size / 10):
        mask, mask_entries = make_mask(offset=offset)
        if shuffle_mask:
            mask = apply_mask_shuffle(m=mask)
        d_std_err = ndimage.standard_deviation(d, labels=mask, index=mask_entries)

        _d_std_err += d_std_err.tolist()

    return np.array(_d_std_err)


def calc_variability_vs_lengthscale(d, dx, retained_target=0.90, method="coarse"):
    Nx, Ny = d.shape
    N = Nx
    assert Nx == Ny

    d_std_err = d.std()
    _out = []
    # for s in range(0, 12):

    if method == "coarse":
        for s in range(0, 12):
            _, d_patch_std_err, _, patch_size = _patch_average_splitting(d=d, s=s)
            # d_patch_std_err = _patch_average(d=d, patch_size=patch_size)
            mean_std_err = np.mean(d_patch_std_err)
            _out.append((patch_size * dx, mean_std_err))
    elif method == "windowed":
        for s in np.linspace(0, 12.0, 48.0):
            patch_size = int(N * 2.0**-s)
            if patch_size / 10 == 0:
                continue
            d_patch_std_err = _patch_average(d=d, patch_size=patch_size)
            mean_std_err = np.mean(d_patch_std_err)
            _out.append((patch_size * dx, mean_std_err))
    else:
        raise NotImplementedError

    _out = np.array(_out)

    patch_size = _out[:, 0]
    variability_retained = _out[:, 1] / d_std_err

    return patch_size, variability_retained


def find_variability_lengthscale(
    d, dx, retained_target=0.90, return_iteration_data=False
):
    """
    Find length-scale over which a given fraction of the domain-wide
    variability is retained
    """
    Nx, Ny = d.shape
    N = Nx
    assert Nx == Ny

    d_std_err = d.std()
    _out = []

    # start with coarse method where domain is split in fractions of 2
    for s in range(0, 12):
        _, d_patch_std_err, _, patch_size = _patch_average_splitting(d=d, s=s)
        mean_std_err = np.mean(d_patch_std_err)

        r = mean_std_err / d_std_err

        if r < retained_target:
            dp_coarse = N / 2 ** (s - 1) - N / 2**s
            break
        else:
            _out.append((patch_size * dx, mean_std_err))

    dp = dp_coarse

    # use finer windowed method until convergence
    while True:
        patch_size += dp

        d_patch_std_err = _patch_average(d=d, patch_size=patch_size)
        mean_std_err = np.mean(d_patch_std_err)

        _out.append((patch_size * dx, mean_std_err))

        r = mean_std_err / d_std_err

        if patch_size < 1:
            raise Exception("Convergence error")

        is_over = dp > 0.0 and r > retained_target
        is_under = dp < 0.0 and r < retained_target

        if is_over or is_under:
            dp = -dp / 2
            if np.abs(dp) == 1:
                break

    if return_iteration_data:
        _out = np.array(_out)
        patch_size = _out[:, 0]
        variability_retained = _out[:, 1] / d_std_err

        return patch_size, variability_retained
    else:
        return patch_size * dx


def get_variability_lengthscale_with_height(
    param_name, var_name, tn, variability_target, z_max
):

    data = utils.get_data(
        model_name=model_name,
        case_name=case_name,
        var_name=var_name,
        tn=tn,
        param_name=param_name,
    )

    z_ = data.zm[np.logical_and(data.zm > 0.0, data.zm <= z_max)]
    dx = np.diff(data.xt).min()

    _arr = []
    for z in tqdm(z_):

        d = (
            data[var_name]
            .isel(time=0, drop=True)
            .where(data.zm == z, drop=True)
            .squeeze()
            .values
        )

        bl_l = find_variability_lengthscale(
            d,
            dx=dx,
            retained_target=variability_target,
        )

        _arr.append((z, bl_l))

    _arr = np.array(_arr)
    z = _arr[:, 0]
    bl_l = _arr[:, 1]

    return z, bl_l


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import argparse

    import matplotlib.pyplot as plot

    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--param_name", nargs="+")
    argparser.add_argument("--tn", nargs="+", type=int)
    argparser.add_argument("--var_name", nargs="+")
    argparser.add_argument("--z_max", default=700.0)
    argparser.add_argument(
        "--variability_target", nargs="+", default=[0.90], type=float
    )

    args = argparser.parse_args()

    combined = len(args.param_name) > 1

    param_name = args.param_name

    if combined:
        plot.figure()

    for param_name in args.param_name:
        for var_name in args.var_name:
            for tn in args.tn:
                if not combined:
                    plot.figure()

                for var_target in args.variability_target:
                    z, bl_l = get_variability_lengthscale_with_height(
                        param_name,
                        var_name,
                        tn=tn,
                        variability_target=var_target,
                        z_max=args.z_max,
                    )

                    if combined:
                        label = "{}%, {}".format(100.0 * var_target, param_name)
                        title = """Varibility length-scale of {} in RICO at t={}min
                                """.format(
                            var_name, tn
                        )
                    else:
                        label = "{}%".format(100.0 * var_target)
                        title = """Varibility length-scale of {} in RICO {} at t={}min
                                """.format(
                            var_name, param_name, tn
                        )
                    plot.plot(bl_l, z, marker="x", label=label)
                    plot.xlabel("length-scale [m]")
                    plot.ylabel("height [m]")

                plot.legend()
                plot.title(title)

                if not combined:
                    fn = "{}__{}__tn{}.pdf".format(
                        param_name.replace("/", "_"), var_name, tn
                    )

                    plot.savefig(fn)
                    print("Plot saved to {}".format(fn))

    if combined:
        fn = "{}_{}_tn{}.pdf".format(param_name.replace("/", "_"), var_name, tn)

        plot.savefig(fn)
        print("Plot saved to {}".format(fn))
