import glob

import xarray as xr

def get_data(base_name, mask_identifier='*'):
    fns = glob.glob("{}.minkowski_scales.{}.nc".format(base_name, mask_identifier))
    fns += glob.glob("{}.objects.{}.integral.*.nc".format(base_name, mask_identifier)) 

    ds = xr.open_mfdataset(fns)

    # 4/3*pi*r**3 => r = (3/4*1/pi*v)**(1./3.)
    ds['r_equiv'] = (3./(4.*3.14)*ds.volume__sum)**(1./3.)
    ds.r_equiv.attrs['units'] = 'm'
    ds.r_equiv.attrs['longname'] = 'equivalent radius'

    return ds

