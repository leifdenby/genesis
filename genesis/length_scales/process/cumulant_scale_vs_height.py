"""
Utility script to process all 3D dataset and calculate cumulant length-scales
in 2D domain-wide cross-sections
"""
import os
import warnings

import xarray as xr
import numpy as np
from tqdm import tqdm

from collections import OrderedDict

from genesis.length_scales import cumulant_analysis


DATA_ROOT = os.environ.get(
    'DATA_ROOT','/nfs/see-fs-02_users/earlcd/datastore/a289/LES_analysis_output/'
)

def z_center_field(phi_da):
    assert phi_da.dims[-1] == 'zm'

    # average vertical velocity to cell centers
    zt_vals = 0.5*(phi_da.zm[1:].values + phi_da.zm[:-1].values)
    zt = xr.DataArray(zt_vals, coords=dict(zt=zt_vals),
                      attrs=dict(units='m'),dims=('zt',))

    # create new coordinates for cell-centered vertical velocity
    coords=OrderedDict(phi_da.coords)
    del(coords['zm'])
    coords['zt'] = zt

    phi_cc_vals = 0.5*(phi_da[...,1:].values + phi_da[...,:-1].values)

    dims = list(phi_da.dims)
    dims[dims.index('zm')] = 'zt'

    phi_cc = xr.DataArray(
        phi_cc_vals, coords=coords, dims=dims,
        attrs=dict(units=phi_da.units, longname=phi_da.longname)
    )

    phi_cc.name = phi_da.name

    return phi_cc

def compute_vertical_flux(phi_da, w_da):
    """
    Compute vertical flux of `phi_da` using vertical velocities `w_da`. Both are
    expected to be xarray.DataArray
    """
    assert phi_da.time == w_da.time
    assert phi_da.dims[-1] == 'zt'
    assert w_da.dims[-1] == 'zm'

    w_cc = z_center_field(phi_da=w_da)

    w_horz_mean = w_cc.mean('x').mean('y')
    v_horz_mean = phi_da.mean('x').mean('y')
    dw = w_cc - w_horz_mean
    dv = phi_da - v_horz_mean

    v_flux = dw*dv
    v_flux.attrs['units'] = "{} {}".format(w_da.units, phi_da.units)
    v_flux.attrs['longname'] = "{} vertical flux".format(phi_da.longname)

    return v_flux

def _extract_vertical(da, z):
    fn = da.from_file

    fn_slice = fn.replace('.nc', '.z{}m.nc'.format(z.values))

    if not os.path.exists(fn_slice):
        da_slice = da.isel(time=0, drop=True)\
                     .where(da.zt==z, drop=True).squeeze()

        da_slice = da_slice.transpose('x', 'y')

        # copy over the shorthand name so that they can be used when naming the
        # cumulant
        da_slice.name = da.name

        da_slice.to_netcdf(fn_slice)
        da_slice.close()

    return xr.open_dataarray(fn_slice, decode_times=False)

def get_height_variation_of_characteristic_scales(v1_3d, v2_3d, z_max,
                                                  z_min=0.0, mask=None):
    datasets = []

    z_ = v1_3d.zt[np.logical_and(v1_3d.zt > z_min, v1_3d.zt <= z_max)]

    for z in tqdm(z_):
        v1 = _extract_vertical(v1_3d, z=z)
        v2 = _extract_vertical(v2_3d, z=z)

        scales = cumulant_analysis.charactistic_scales(v1=v1, v2=v2, mask=mask)

        datasets.append(scales)

    d = xr.concat(datasets, dim='zt')

    # since these were all computations for the same cumulant we can set just
    # one value which will be used for all. This will allow merge across
    # cumulants later
    d['cumulant'] = d.cumulant.values[0]

    return d


def process(model_name, case_name, param_names, variable_sets, z_min, z_max, tn,
            mask=None):
    datasets = []

    for param_name in param_names:
        param_datasets = []

        for var_name_1, var_name_2 in variable_sets:
            v1_3d = get_data(model_name, case_name, param_name,
                             var_name=var_name_1, tn=tn)

            if var_name_2 != var_name_2:
                v2_3d = get_data(model_name, case_name, param_name,
                                 var_name=var_name_2, tn=tn)
            else:
                v2_3d = v1_3d


            characteristic_scales = get_height_variation_of_characteristic_scales(
                v1_3d=v1_3d, v2_3d=v2_3d, z_max=z_max, z_min=z_min, mask=mask
            )

            characteristic_scales['dataset_name'] = "{}__{}".format(
                case_name, param_name, mask=mask)
            param_datasets.append(characteristic_scales)

        dataset_param = xr.concat(param_datasets, dim='cumulant')
        # all cumulant calculations were for the same dataset, set one value
        # so we can concat later across `dataset`
        dataset_param['dataset_name'] = dataset_param.dataset_name.values[0]

        datasets.append(dataset_param)

    all_data = xr.concat(datasets, dim='dataset_name')


    return all_data


def get_fn(model_name, case_name, param_name, var_name, tn):
    return os.path.join(DATA_ROOT, model_name, case_name, param_name, 
                        '3d_blocks', 'full_domain',
                        '{}.tn{}.{}.nc'.format(case_name, tn, var_name))


def _check_coords(da):
    if not 'x' in da.coords or not 'y' in da.coords:
        warnings.warn("Coordinates for x and y are missing input, assuming "
                      "dx=25m for now, but these files need regenerating "
                      "after EGU")
        dx = 25.0

        da['x'] = dx*np.arange(len(da.x)) - dx*len(da.x)/2
        da['y'] = dx*np.arange(len(da.y)) - dx*len(da.y)/2

    return da

def get_data(model_name, case_name, param_name, var_name, tn):
    compute_flux = False

    fn = get_fn(model_name, case_name, param_name, var_name, tn)
    if var_name.endswith('_flux') and not os.path.exists(fn):
        var_name = var_name.replace('_flux', '')
        compute_flux = True
    try:
        phi_da = xr.open_dataarray(fn, decode_times=False,
                                   chunks=dict(zt=20))
        phi_da = phi_da.rename(dict(xt='x', yt='y'))
        phi_da.name = var_name
        print("Using {}".format(fn))
    except IOError:
        print fn
        raise


    if var_name == 'w':
        phi_da = z_center_field(phi_da=phi_da)

    # so we can store horizontal slices later
    phi_da.attrs['from_file'] = os.path.realpath(fn)


    if not compute_flux:
        return _check_coords(phi_da)
    else:
        fn_w = get_fn(model_name, case_name, param_name, 'w', tn)
        w_da = xr.open_dataarray(fn_w, decode_times=False,)
        w_da = w_da.rename(dict(xt='x', yt='y'))

        phi_flux_da = compute_vertical_flux(phi_da=phi_da, w_da=w_da)
        phi_flux_da.name = "{}_flux".format(var_name)

        return _check_coords(phi_flux_da)


def get_cross_section(model_name, case_name, var_name, z, tn,
                      param_name, z_max=700., method=None):

    ar = get_data(model_name, case_name, param_name, var_name=var_name, tn=tn)

    da = ar.sel(zt=z, drop=True, method=method).squeeze()

    da = da.transpose('x', 'y')

    da['zt'] = ((), z, dict(units='m'))

    return da


def run_default():
    model = 'uclales'
    case_name = 'rico'

    VARIABLE_SETS = (
        ('w', 'w'),
        ('q', 'q'),
        ('t', 't'),
        ('l', 'l'),
        ('q_flux', 'q_flux'),
        ('t_flux', 't_flux'),
        ('l_flux', 'l_flux'),
    )

    PARAM_NAMES = [
        'fixed_flux_shear/nx800',
        'fixed_flux_noshear/nx800',
    ]

    return process(model_name, case_name, PARAM_NAMES, VARIABLE_SETS,
                   z_min=0., z_max=700., tn=3)

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    DEFAULT_VARS = "w q t l q_flux t_flux l_flux".split(" ")

    argparser.add_argument('case_name', help='e.g. `rico`', type=str)
    argparser.add_argument('--param_name',
        help='e.g. `fixed_flux_shear/nx800`', type=str, default=['',],
        nargs='+')
    argparser.add_argument('tn', type=int)
    argparser.add_argument('--vars', default=DEFAULT_VARS, nargs="+")
    argparser.add_argument('--model', default='uclales')
    argparser.add_argument('--z_max', default=700., type=float)
    argparser.add_argument('--z_min', default=0., type=float)
    argparser.add_argument('--mask-name', default=None, type=str)
    argparser.add_argument('--mask-field', default=None, type=str)
    argparser.add_argument('--invert-mask', default=False, action="store_true")

    args = argparser.parse_args()

    if args.model != 'uclales':
        raise NotImplementedError


    out_filename = "{}.tn{}.cumulant_length_scales.nc".format(
        args.case_name, args.tn
    )

    if not args.mask_name is None:
        if args.mask_field is None:
            mask_field = args.mask_name
            mask_description = args.mask_name
        else:
            mask_field = args.mask_field
            mask_description = "{}__{}".format(args.mask_name, args.mask_field)

        input_name = '{}.tn{}'.format(args.case_name, args.tn)
        fn_mask = "{}.{}.mask.nc".format(input_name, args.mask_name)
        if not os.path.exists(fn_mask):
            raise Exception("Can't find mask file `{}`".format(fn_mask))

        ds_mask = xr.open_dataset(fn_mask, decode_times=False)
        if not mask_field in ds_mask:
            raise Exception("Can't find `{}` in mask, loaded mask file:\n{}"
                            "".format(mask_field, str(ds_mask)))
        else:
            mask = ds_mask[mask_field]

        if args.invert_mask:
            mask_attrs = mask.attrs
            mask = ~mask
            mask.name = 'not_{}'.format(mask.name)
            out_filename = out_filename.replace(
                '.nc', '.masked.not__{}.nc'.format(mask_description)
            )
            mask.attrs.update(mask_attrs)
        else:
            out_filename = out_filename.replace(
                '.nc', '.masked.{}.nc'.format(mask_description)
            )
    else:
        mask = None


    variable_sets = zip(args.vars, args.vars)

    data = process(
        model_name=args.model,
        case_name=args.case_name,
        param_names=args.param_name,
        variable_sets=variable_sets,
        z_min=args.z_min,
        z_max=args.z_max,
        tn=args.tn,
        mask=mask
    )

    data.to_netcdf(out_filename, mode='w')
    print("Output written to {}".format(out_filename))
