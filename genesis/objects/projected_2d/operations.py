import numpy as np
import dask_image.ndmeasure as dmeasure


class CloudType:
    PASSIVE = 1
    SINGLE_PULSE = 2
    OUTFLOW = 3
    ACTIVE = 4


def present(ds, t0):
    """
    Return clouds that are present at time `t0`
    """
    # time of appearance
    tmin = ds.smcloudtmin
    # time of disappearance
    tmax = ds.smcloudtmax

    m = (tmin <= t0) & (t0 <= tmax)

    return m


def cloud_type(ds):
    return ds.smcloudtype


def cloud_id(ds):
    return ds.smcloudid


def cloudbase_max_height_by_histogram_peak(ds, t0, da_nrcloud, da_cldbase, dx):
    """Find the maximum height of points which are consider to part of the
    cloud base. This height is found by looking at the histogram of
    cloud-underside heights, picking the first peak and including all points
    that are within twice the height the height from the base to the peak. If a
    second peak is found within this range the cloud is deemed to have a
    multi-leveled cloudbase and is excluded. Also clouds which have too few
    vertical levels so that the double distance would imply that the entire
    cloud is the base are excluded."""

    class InSuficientPointsException(Exception):
        pass

    class TopheavyCloudException(Exception):
        pass

    class MultilevelCloudbaseException(Exception):
        pass

    z_clb_points = da_cldbase.sel(time=t0).squeeze()
    nrcloud = da_nrcloud.sel(time=t0).squeeze().astype(int)

    z_min = z_clb_points.min().item()
    z_max = z_clb_points.max().item()
    hist_range = (z_min - dx * 0.5, z_max + dx * 0.5)
    nbins = int((hist_range[1] - hist_range[0]) / dx)
    z_bins_c = np.arange(z_min, z_max, nbins)

    fn_unique_dropna = lambda v: np.unique(v.data[~np.isnan(v.data)])
    cloud_ids = fn_unique_dropna(nrcloud)[1:]

    histogram = dmeasure.histogram(
        image=z_clb_points,
        min=hist_range[0],
        max=hist_range[1],
        bins=nbins,
        label_image=nrcloud,
        index=cloud_ids,
    ).compute()

    z_base_max = np.zeros(cloud_ids.shape)
    for n, c_id in enumerate(cloud_id(ds)):
        try:
            bin_counts = histogram[n]

            # insert zero number of cells at ends so that ends can be peaks too
            n_ext = np.append(np.insert(bin_counts, 0, 0), 0)

            is_peak = np.logical_and(n_ext[0:-2] < n_ext[1:-1], n_ext[1:-1] > n_ext[2:])

            peaks = np.nonzero(is_peak)[0]

            if len(peaks) == 0:
                raise InSuficientPointsException(
                    "Couldn't find a peak in the" " cldbase points histogram"
                )

            first_peak = peaks[0]

            # include up to twice the peak index height as part of the cloud base
            try:
                z_base_lim = z_bins_c[first_peak * 2]
            except IndexError:
                raise TopheavyCloudException

            # Is there a second peak nearby?
            if len(peaks) > 1 and first_peak * 2 > peaks[1]:
                raise MultilevelCloudbaseException

            z_base_max[n] = z_base_lim

        except (
            InSuficientPointsException,
            TopheavyCloudException,
            MultilevelCloudbaseException,
        ) as e:
            z_base_max[n] = np.nan

    return z_base_max
