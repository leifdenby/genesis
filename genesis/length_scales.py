"""
Methods for quantifying coherent length-scales in the convection boundary-layer
"""
import numpy as np
from scipy import ndimage

def _patch_average_splitting(d, s, shuffle_mask=False):
    """
    Split data d into 2**s patches and calculate mean and standard deviation
    over all patches
    """
    Nx, Ny = d.shape
    N = Nx
    assert Nx == Ny

    i, j = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')

    def make_mask(s):
        m = 2**s
        patch_size = N/m
        mask = (i / (patch_size)) + m*(j / (patch_size))
        if np.any(np.isnan(mask)):
            raise Exception("Can't make mask of with this splitting")
        return mask, np.unique(mask), patch_size

    def apply_mask_shuffle(m):
        m_flat = m.reshape(Nx*Ny)
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

    i, j = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')

    def make_mask(offset=0):
        """
        Make a mask with regions labelled in consecutive patches, remove any
        patches which because of offset window are outside of domain
        """
        assert offset < patch_size
        m = (N-offset) / patch_size
        mask = ((i - offset) / (patch_size)) + m*((j - offset) / (patch_size))
        if np.any(np.isnan(mask)):
            raise Exception("Can't make mask of with this splitting")

        mask[m*patch_size+offset:,:] = -1
        mask[:,m*patch_size+offset:] = -1
        mask[:offset,:] = -1
        mask[:,:offset] = -1

        return mask, np.unique(mask)

    def apply_mask_shuffle(m):
        m_flat = m.reshape(Nx*Ny)
        np.random.shuffle(m_flat)
        return m_flat.reshape(Nx, Ny)

    _d_std_err = []

    for offset in range(0, patch_size, patch_size/10):
        mask, mask_entries = make_mask(offset=offset)
        if shuffle_mask:
            mask = apply_mask_shuffle(m=mask)
        d_std_err = ndimage.standard_deviation(d, labels=mask, index=mask_entries)

        _d_std_err += d_std_err.tolist()

    return np.array(_d_std_err)


def calc_variability_vs_lengthscae(d, dx, retained_target=0.90,
                                   method='coarse'):
    Nx, Ny = d.shape
    N = Nx
    assert Nx == Ny

    d_std_err = d.std()
    _out = []
    # for s in range(0, 12):

    if method == 'coarse':
        for s in range(0, 12):
            _, d_patch_std_err, _, patch_size = _patch_average_splitting(d=d, s=s)
            # d_patch_std_err = _patch_average(d=d, patch_size=patch_size)
            mean_std_err = np.mean(d_patch_std_err)
            _out.append((patch_size*dx, mean_std_err))
    elif method == 'windowed':
        for s in np.linspace(0, 12.0, 48.0):
            patch_size = int(N*2.0**-s)
            if patch_size/10 == 0:
                continue
            d_patch_std_err = _patch_average(d=d, patch_size=patch_size)
            mean_std_err = np.mean(d_patch_std_err)
            _out.append((patch_size*dx, mean_std_err))
    else:
        raise NotImplementedError


    _out = np.array(_out)

    patch_size = _out[:,0]
    variability_retained = _out[:,1]/d_std_err

    return patch_size, variability_retained

def find_variability_lengthscale(d, dx, retained_target=0.90,
                                 return_iteration_data=False):
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

        r = mean_std_err/d_std_err

        if r < retained_target:
            dp_coarse = N/2**(s-1) - N/2**s
            break
        else:
            _out.append((patch_size*dx, mean_std_err))

    dp = dp_coarse

    # use finer windowed method until convergence
    while True:
        patch_size += dp

        d_patch_std_err = _patch_average(d=d, patch_size=patch_size)
        mean_std_err = np.mean(d_patch_std_err)

        _out.append((patch_size*dx, mean_std_err))

        r = mean_std_err/d_std_err

        if patch_size < 1:
            raise Exception("Convergence error")

        is_over = dp > 0.0 and r > retained_target
        is_under = dp < 0.0 and r < retained_target

        if is_over or is_under:
            dp = -dp/2
            if np.abs(dp) == 1:
                break

    if return_iteration_data:
        _out = np.array(_out)
        patch_size = _out[:,0]
        variability_retained = _out[:,1]/d_std_err

        return patch_size, variability_retained
    else:
        return patch_size*dx
