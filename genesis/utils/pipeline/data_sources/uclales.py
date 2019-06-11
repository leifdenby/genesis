import xarray as xr
import numpy as np

from pathlib import Path
import os
from ...calc_flux import z_center_field

FIELD_NAME_MAPPING = dict(
    w='w_zt',
    xt='xt',
    yt='yt',
    zt='zt',
    qv='q',
    qc='l',
    qr='r',
    theta_l='t',
    cvrxp='cvrxp',
    p='p',
)

FIELD_DESCRIPTIONS = dict(
    w='vertical velocity',
    qv='water vapour',
    qc='cloud liquid water',
    theta='potential temperature',
    cvrxp='radioactive tracer',
    theta_l='liquid potential temperature',
)

UNITS_FORMAT = {
    'METERS/SECOND': 'm/s',
    'KELVIN': 'K',
    'KG/KG': 'kg/kg',
}

DERIVED_FIELDS = dict(
    T=('qc', 'theta_l', 'p'),
)

FN_FORMAT_3D = "3d_blocks/full_domain/{experiment_name}.tn{timestep}.{field_name}.nc"

def _get_uclales_field(field_name):
    field_name_src = FIELD_NAME_MAPPING.get(field_name)

    if field_name_src is None:
        raise NotImplementedError("please define a mapping for the field `{}`"
                                  " in {}".format(field_name, __file__))

    return field_name_src

def _get_uclales_field_description(field_name):
    field_description = FIELD_DESCRIPTIONS.get(field_name)

    if field_description is None:
        raise NotImplementedError("please define a description for the field "
                                  "`{}` in {}".format(field_name, __file__))

    return field_description

def _scale_field(da):
    if da.units == 'km':
        da.values *= 1000.
        da.attrs['units'] = 'm'
    elif 'q' in da.name and da.units == 'kg/kg':
        da.values *= 1000.
        da.attrs['units'] = 'g/kg'

    return da

def _cleanup_units(units):
    return UNITS_FORMAT.get(units, units)

def _calculate_theta_v(theta, qv):
    assert qv.units == 'g/kg'
    assert theta.units == 'K'

    return theta*(1.0 + 0.61*qv/1000.)

def extract_field_to_filename(dataset_meta, path_out, field_name, **kwargs):
    field_name_src = _get_uclales_field(field_name)

    fn_format = dataset_meta.get('fn_format', FN_FORMAT_3D)
    path_in = Path(dataset_meta['path'])/fn_format.format(
        field_name=field_name_src, **dataset_meta
    )

    if field_name_src == 'w_zt':
        path_in = path_in.parent/path_in.name.replace('.w_zt.', '.w.')
        da_w_orig = xr.open_dataarray(path_in, decode_times=False)
        da = z_center_field(da_w_orig)
    else:
        da = xr.open_dataarray(path_in, decode_times=False)

    can_symlink = True

    for c in "xt yt zt".split(" "):
        if 'fixes' in dataset_meta:
            if 'missing_{}_coordinate'.format(c[0]) in dataset_meta['fixes']:
                can_symlink = False

                dx = dataset_meta['dx']
                da.coords[c] = -0.5*dx + dx*np.arange(0, len(da[c]))
                da.coords[c].attrs['units'] = 'm'

    # the CF conventions stipulate that the long name attribute should be named
    # `long_name` not `longname`
    da.attrs['long_name'] = da.longname

    if field_name_src != field_name:
        can_symlink = False
        da.name = field_name

    if can_symlink:
        os.symlink(str(path_in), str(path_out))
    else:
        da.to_netcdf(path_out)

    # if field_name == 'theta_v':
        # assert 'qv' in kwargs and 't' in kwargs

        # da_theta = kwargs['t'].open()
        # da_qv = kwargs['qv'].open()

        # da = _calculate_theta_v(theta=da_theta, qv=da_qv)
        # da.name = 'theta_v'
        # da.attrs['units'] = 'K'
        # da.attrs['long_name'] = 'virtual potential temperature'
    # else:
        # ds = xr.open_dataset(path_in)


        # da = ds[field_name_src]
        # if field_name == 'w_zt':
            # field_name = 'w'
        # da.name = field_name

        # new_coords = "xt yt".split(" ")
        # old_coords = [_get_uclales_field(c) for c in new_coords]
        # coord_map = dict(zip(old_coords, new_coords))
        # da = da.rename(coord_map)
        # da.attrs['long_name'] = _get_uclales_field_description(field_name)
        # da.attrs['units'] = _cleanup_units(da.units)

        # # all height levels should be the same, we do a quick check and then
        # # use those values
        # z__min = ds.VLEV.min(dim=old_coords)
        # z__max = ds.VLEV.max(dim=old_coords)
        # assert np.all(z__max - z__min < 1.0e-10)
        # da['zt'] = np.round(z__min, decimals=3)
        # da.zt.attrs['units'] = ds[old_coords[0]].units

        # da = da.swap_dims(dict(vertical_levels='zt'))

        # for c in "xt yt zt".split(" "):
            # da.coords[c] = _scale_field(da[c])

        # _scale_field(da)

        # da = da.squeeze()

    # da.to_netcdf(path_out)
