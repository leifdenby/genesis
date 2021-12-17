"""
functions for deriving more variables than the cloud-tracking code currently
calculates
"""
import dask_image.ndmeasure as dmeasure
import numpy as np
import xarray as xr


def per_object_histogram(da_nrobj, da, bin_size):
    """
    Create a per-object (given by labels in `da_nrobj`) histogram for the
    variable `da` with bin-size `bin_size`
    """
    da = da.squeeze()
    da_nrobj = da_nrobj.squeeze().astype(int)

    assert da_nrobj.dims == da.dims
    assert da_nrobj.shape == da.shape

    v_min = da.min().item()
    v_max = da.max().item()
    hist_range = (v_min - bin_size * 0.5, v_max + bin_size * 0.5)
    nbins = int((hist_range[1] - hist_range[0]) / bin_size)
    v_bins_c = np.linspace(v_min, v_max, nbins)

    def fn_unique_dropna(v):
        return np.unique(v.data[~np.isnan(v.data)])

    object_ids = fn_unique_dropna(da_nrobj)[1:]

    histogram = dmeasure.histogram(
        image=da,
        min=hist_range[0],
        max=hist_range[1],
        bins=nbins,
        label_image=da_nrobj,
        index=object_ids,
    ).compute()

    # I am not really sure why, but I need to cast to a list here before I can
    # get a 2D array
    histogram = np.array(histogram.tolist())

    da_bins = xr.DataArray(v_bins_c, dims=da.name, attrs=da.attrs)

    return xr.DataArray(
        histogram,
        dims=("object_id", da.name),
        coords={"object_id": object_ids, da.name: da_bins},
        attrs=dict(
            long_name=f"{da.long_name} count",
            units="1",
        ),
    )


def cloudbase_max_height_by_histogram_peak(da_nrcloud, da_cldbase, dx):
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

    raise NotImplementedError


#    z_clb_points = da_cldbase.squeeze()
#    nrcloud = da_nrcloud.squeeze().astype(int)
#
#    assert nrcloud.time == z_clb_points.time
#
#    z_base_max = np.zeros(object_ids.shape)
#    for n, c_id in enumerate(cloud_ids):
#        try:
#            bin_counts = histogram[n]
#
#            # insert zero number of cells at ends so that ends can be peaks too
#            n_ext = np.append(np.insert(bin_counts, 0, 0), 0)
#
#            is_peak = np.logical_and(n_ext[0:-2] < n_ext[1:-1], n_ext[1:-1] > n_ext[2:])
#
#            peaks = np.nonzero(is_peak)[0]
#
#            if len(peaks) == 0:
#                raise InSuficientPointsException(
#                    "Couldn't find a peak in the" " cldbase points histogram"
#                )
#
#            first_peak = peaks[0]
#
#            # include up to twice the peak index height as part of the cloud base
#            try:
#                z_base_lim = z_bins_c[first_peak * 2]
#            except IndexError:
#                raise TopheavyCloudException
#
#            # Is there a second peak nearby?
#            if len(peaks) > 1 and first_peak * 2 > peaks[1]:
#                raise MultilevelCloudbaseException
#
#            z_base_max[n] = z_base_lim
#
#        except (
#            InSuficientPointsException,
#            TopheavyCloudException,
#            MultilevelCloudbaseException,
#        ):
#            z_base_max[n] = np.nan
#
#    return xr.DataArray(
#        z_base_max,
#        dims="cloud_id",
#        coords=dict(cloud_id=cloud_ids),
#        attrs=dict(
#            long_name="Cloud-base height (from underside height dist peak)",
#            units=da_cldbase.units,
#        ),
#    )
