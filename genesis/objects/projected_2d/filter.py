

def present(ds_tracking, t0):
    """
    Return clouds that are present at time `tn`
    """
    # time of appearance
    tmin = ds_tracking.smcloudtmin
    # time of disappearance
    tmax = ds_tracking.smcloudtmax

    m = (tmin <= t0) & (t0 <= tmax)

    return ds_tracking.where(m, drop=True)

def cloud_type(ds_tracking, types):
    return ds_tracking.sel(smcloud=[
        t.value for t in types
    ]
