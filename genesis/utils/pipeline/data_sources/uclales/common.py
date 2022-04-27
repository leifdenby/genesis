import numpy as np


def _fix_time_units(da):
    modified = False
    if np.issubdtype(da.dtype, np.datetime64):
        # already converted since xarray has managed to parse the time in
        # CF-format
        pass
    elif da.attrs["units"].startswith("seconds since 2000-01-01"):
        # I fixed UCLALES to CF valid output, this is output from a fixed
        # version
        pass
    elif da.attrs["units"] == "s":
        # the cross-section files output by UCLALES don't have a reference time
        # for the time units
        da.attrs["units"] = "seconds since 2000-01-01 00:00:00"
        modified = True
    elif da.attrs["units"].startswith("seconds since 2000-00-00"):
        da.attrs["units"] = da.attrs["units"].replace(
            "seconds since 2000-00-00",
            "seconds since 2000-01-01",
        )
        modified = True
    elif da.attrs["units"].startswith("seconds since 0-00-00"):
        # 2D fields have strange time units...
        da.attrs["units"] = da.attrs["units"].replace(
            "seconds since 0-00-00",
            "seconds since 2000-01-01",
        )
        modified = True
    elif da.attrs["units"].startswith("seconds since 0-0-0"):
        # 2D fields have strange time units...
        da.attrs["units"] = da.attrs["units"].replace(
            "seconds since 0-0-0",
            "seconds since 2000-01-01",
        )
        modified = True
    elif da.attrs["units"] == "day as %Y%m%d.%f":
        da = (da * 24 * 60 * 60).round().astype(int)
        da.attrs["units"] = "seconds since 2000-01-01 00:00:00"
        modified = True
    else:
        raise NotImplementedError(da.attrs["units"])
    return da, modified
