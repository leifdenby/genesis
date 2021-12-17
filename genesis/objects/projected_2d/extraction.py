import numpy as np

from . import CloudType
from .object_set import ObjectSet
from .utils import create_mask_from_object_set


def get_cloud_underside_for_new_formed_clouds(
    ds_tracking, da_cldbase_2d, t0, t_age_max
):
    """
    Extract the cloud-underside height from time-dependent 2D cloud-underside
    field `da_cldbase_2d` at time `t0` by identifying clouds 2D in tracking
    data (`ds_tracking`) which are at max `t_age_max` seconds old.
    """
    object_set = ObjectSet(ds=ds_tracking)

    if not np.issubdtype(np.array(t_age_max).dtype, np.timedelta64):
        t_age_max = np.timedelta64(int(t_age_max), "s")

    # clouds that are going to do vertical transport
    object_set = object_set.filter(
        object_type__in=[CloudType.SINGLE_PULSE, CloudType.ACTIVE]
    )

    object_set = object_set.filter(present=True, kwargs=dict(t0=t0))

    # the maximum height we allow is given distance above the domain-wide
    # minimum height
    # z_base_max = da_cldbase_2d.min().item() + dz_base_max

    # avoid mid-level convection clouds
    # object_set = object_set.filter(
    #   # cloudbase_max_height_by_histogram_peak__lt=z_base_max,
    #   # kwargs=dict(t0=t0, dx=dz, da_cldbase=da_cldbase_2d),
    # )

    # remove clouds that are more than 3min old
    object_set = object_set.filter(age__lt=t_age_max, kwargs=dict(t0=t0))

    da_mask_objects = create_mask_from_object_set(
        ds_tracking=ds_tracking, object_set=object_set, t0=t0
    )

    da_cldbase_heights_2d = da_cldbase_2d.sel(time=t0).where(~da_mask_objects.isnull())

    da_cldbase_heights_2d.attrs["num_clouds"] = len(object_set)

    return da_cldbase_heights_2d


def get_approximate_cloud_underside(qc, z_tol=100.0):
    """
    Using 3D condensate field `qc` create a 2D array representing the underside
    height of the lowest cloudy point per column
    """
    z_cloud_underside = qc.zt.where(qc > 0.0).min(dim="zt")

    m = z_cloud_underside < z_tol + z_cloud_underside.min()
    z_cb = z_cloud_underside.where(m)

    return z_cb
