"""
Calculate new properties of 2D projected objects by aggregating over their
individual properties in their 2D extent
"""
import numpy as np
import xarray as xr


class InSuficientPointsException(Exception):
    pass


class TopheavyCloudException(Exception):
    pass


def _extract_single_cloud_cloudbase(da_cldbase_histogram):
    # only consider the bins where we have counts
    da_ = da_cldbase_histogram.where(da_cldbase_histogram > 0, drop=True)

    bin_counts = da_.values
    # insert zero number of cells at ends so that ends can be peaks too
    n_ext = np.append(np.insert(bin_counts, 0, 0), 0)

    is_peak = np.logical_and(n_ext[0:-2] < n_ext[1:-1], n_ext[1:-1] > n_ext[2:])

    peaks = np.nonzero(is_peak)[0]

    if len(peaks) == 0:
        raise InSuficientPointsException(
            "Couldn't find a peak in the" " cldbase points histogram"
        )

    first_peak = peaks[0]

    z_cb = da_.isel(cldbase=first_peak).cldbase

    # the peak should be in the bottom half of the cloud
    z_cloudunderside_max = da_.cldbase.max()
    z_cloudunderside_min = da_.cldbase.min()
    z_cloudunderside_height = z_cloudunderside_max - z_cloudunderside_min

    if (z_cb - z_cloudunderside_min) > 0.5 * z_cloudunderside_height:
        raise TopheavyCloudException

    return z_cb


def cloudbase_max_height_by_histogram_peak(da_cldbase_histogram):
    """Find the maximum height of points which are consider to part of the
    cloud base. This height is found by looking at the histogram of
    cloud-underside heights, picking the first peak. Clouds for which a peak in
    the histogram can't be found are excluded"""

    def fn(da_cldbase_single_cloud):
        try:
            z_cb = _extract_single_cloud_cloudbase(da_cldbase_single_cloud)
        except (InSuficientPointsException, TopheavyCloudException):
            z_cb = np.nan

        dims = list(da_cldbase_single_cloud.dims)
        dims.remove("cldbase")
        coords = {d: da_cldbase_single_cloud[d] for d in dims}
        return xr.DataArray(
            [
                z_cb,
            ],
            coords=coords,
            dims=dims,
        )

    return da_cldbase_histogram.groupby("object_id").apply(fn)
