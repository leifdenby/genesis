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
    theta_l_v='theta_l_v',
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
    theta_l_v=('theta_l', 'qv', 'qc', 'qr')
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
    modified = False
    if da.units == 'km':
        da.values *= 1000.
        da.attrs['units'] = 'm'
        modified = True
    elif da.name.startswith('q'):
        assert da.max() < 1.0 and da.units == 'g/kg'
        # XXX: there is a bug in UCLALES where the units reported (g/kg) are
        # actually wrong (they are kg/kg), this messes up the density
        # calculation later if we don't fix it
        da.values *= 1000.
        modified = True
    return da, modified

def _cleanup_units(units):
    return UNITS_FORMAT.get(units, units)

def _fix_long_name(da):
    modified = False
    # the CF conventions stipulate that the long name attribute should be named
    # `long_name` not `longname`
    if 'longname' in da.attrs and not 'long_name' in da.attrs:
        da.attrs['long_name'] = da.longname
        modified = True
    return da, modified

def extract_field_to_filename(dataset_meta, path_out, field_name, **kwargs):
    field_name_src = _get_uclales_field(field_name)

    fn_format = dataset_meta.get('fn_format', FN_FORMAT_3D)
    path_in = Path(dataset_meta['path'])/fn_format.format(
        field_name=field_name_src, **dataset_meta
    )

    can_symlink = True

    if field_name_src == 'w_zt':
        path_in = path_in.parent/path_in.name.replace('.w_zt.', '.w.')
        da_w_orig = xr.open_dataarray(path_in, decode_times=False)
        da_w_orig, _ = _fix_long_name(da_w_orig)
        da = z_center_field(da_w_orig)
        can_symlink = False
    elif field_name == 'theta_l_v':
        da = _calc_theta_l_v(**kwargs)
        can_symlink = False
    else:
        da = xr.open_dataarray(path_in, decode_times=False)

    da, modified = _scale_field(da)
    if modified:
        can_symlink = False
    da, modified = _fix_long_name(da)
    if modified:
        can_symlink = False

    for c in "xt yt zt".split(" "):
        if 'fixes' in dataset_meta:
            if 'missing_{}_coordinate'.format(c[0]) in dataset_meta['fixes']:
                can_symlink = False

                dx = dataset_meta['dx']
                da.coords[c] = -0.5*dx + dx*np.arange(0, len(da[c]))
                da.coords[c].attrs['units'] = 'm'

    if field_name_src != field_name:
        can_symlink = False
        da.name = field_name

    if can_symlink and path_in.exists():
        os.symlink(str(path_in), str(path_out))
    else:
        da.to_netcdf(path_out)


def _calc_theta_l_v(theta_l, qv, qc, qr):
    assert qv.units.lower() == 'g/kg' and qv.max() > 1.0
    assert theta_l.units == 'K'

    # qc here refers to q "cloud", the total condensate is qc+qr (here
    # forgetting about ice...)
    eps = 0.608
    theta_l_v = theta_l*(1.0 + 1.0e-3*(eps*qv - (qc+qr)))

    theta_l_v.attrs['units'] = 'K'
    theta_l_v.attrs['long_name'] = 'virtual liquid potential temperature'
    theta_l_v.name = "theta_l_v"
    return theta_l_v
