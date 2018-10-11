import glob

import xarray as xr

def get_data(base_name, mask_identifier='*'):
    glob_patterns = [
        "{}.minkowski_scales.{}.nc".format(base_name, mask_identifier),
        "{}.objects.{}.integral.*.nc".format(base_name, mask_identifier)
    ]

    fns = reduce(lambda a, s: glob.glob(s) + a, glob_patterns, [])

    if len(fns) == 0:
        raise Exception("No files found with glob patterns: {}".format(
                        ", ".join(glob_patterns)))

    ds = xr.open_mfdataset(fns)

    # 4/3*pi*r**3 => r = (3/4*1/pi*v)**(1./3.)
    ds['r_equiv'] = (3./(4.*3.14)*ds.volume__sum)**(1./3.)
    ds.r_equiv.attrs['units'] = 'm'
    ds.r_equiv.attrs['longname'] = 'equivalent radius'

    return ds

