"""
Produce cross-correlation contour plots as function of height and at
cloud-base.  Regions of highest density percentile are contoured
"""
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

import copy
import os
import warnings

import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

from cloud_tracking_analysis import CloudData, CloudType, cloud_operations
from cloud_tracking_analysis.cloud_mask_methods import cloudbase as get_cloudbase_mask
from cloud_tracking_analysis.cloud_mask_methods import CloudbaseEstimationMethod

from . import get_dataset
from ..utils.plot_types import joint_hist_contoured, JointHistPlotError


Z_LEVELS_DEFAULT = np.arange(12.5, 650., 100.)

def get_cloudbase_data(cloud_data, t0, t_age_max=200., z_base_max=700.):
    tn = int(cloud_data.find_closest_timestep(t=t0))

    # clouds that are going to do vertical transport
    cloud_set = cloud_data.all_clouds.filter(
        cloud_type__in=[CloudType.SINGLE_PULSE, CloudType.ACTIVE],
    ).filter(present=True, _tn=tn)

    # avoid mid-level convection clouds
    cloud_set = cloud_set.filter(
        cloudbase_max_height_by_histogram_peak__lt=z_base_max, _tn=tn
    )

    # remove clouds that are more than 3min old
    cloud_set = cloud_set.filter(cloud_age__lt=t_age_max, _tn=tn)

    nrcloud_cloudbase = get_cloudbase_mask(
        cloud_set=cloud_set, tn=tn, method=CloudbaseEstimationMethod.DEFAULT
    )

    cldbase = cloud_set.cloud_data.get('cldbase', tn=tn)
    m = nrcloud_cloudbase == 0
    cldbase_heights_2d = np.ma.masked_array(cldbase, m)

    z_slice = cldbase_heights_2d - cloud_data.dx
    theta_l__belowcloud = cloud_data.get_from_3d(var_name='t', z=z_slice, t=t0)
    # r_l__belowcloud = cloud_data.get_from_3d(var_name='l', z=z_slice, t=tn*60.)
    r_t__belowcloud = cloud_data.get_from_3d(var_name='q', z=z_slice, t=t0)
    dx = cloud_set.cloud_data.dx
    w__belowcloud = cloud_data.get_from_3d(var_name='w', z=z_slice+dx/2., t=t0)

    ds = xr.Dataset()
    # XXX: using non-xarray indexing here, this could be made faster (and
    # probably more robust too)
    if isinstance(r_t__belowcloud, xr.DataArray):
        r_t__belowcloud = r_t__belowcloud.squeeze()
        theta_l__belowcloud = theta_l__belowcloud.squeeze()

    ds['r_t'] = r_t__belowcloud.values[~m]
    ds['theta_l'] = theta_l__belowcloud.values[~m]
    ds['w'] = w__belowcloud.values[~m]

    return ds

def main(ds_3d, ds_cb, z_levels):
    colors = iter(sns.color_palette("cubehelix", len(z_levels)))
    sns.set_color_codes()

    normed_levels = [5, 95, ]  # percentiles to plot in contours

    lines = []

    v1, v2 = ds_3d.data_vars.keys()

    if v1 == 'q' and ds_3d[v1].units == 'g/kg':
        warnings.warn("Scaling variable `q` by 1000 since UCLALES "
                      "incorrectly states the units as g/kg even "
                      "though they are in fact in kg/kg")
        xscale = 1000.
    else:
        xscale = 1.0

    if v2 == 'q' and ds_3d[v2].units == 'g/kg':
        warnings.warn("Scaling variable `q` by 1000 since UCLALES "
                      "incorrectly states the units as g/kg even "
                      "though they are in fact in kg/kg")
        yscale = 1000.
    else:
        yscale = 1.

    for z in tqdm.tqdm(z_levels):
        ds_ = ds_3d.sel(zt=z, method='nearest').squeeze()

        c = next(colors)
        try:
            xd=ds_[v1].values.flatten()*xscale
            yd=ds_[v2].values.flatten()*yscale

            _, _, cnt = joint_hist_contoured(
                xd=xd, yd=yd,
                normed_levels=normed_levels
            )
            for n, l in enumerate(cnt.collections):
                l.set_color(c)
                if n == 0:
                    l.set_label("z={}m".format(ds_.zt.values))
                    lines.append(l)
            pass
        except JointHistPlotError:
            print("error", ds_.zt.values, "skipping")
        except Exception as e:
            print("error", ds_.zt.values)
            raise

    cb_mapping = dict(
        q='r_t', t='theta_l', w='w'
    )

    if not ds_cb is None:
        if not v1 in cb_mapping or not v2 in cb_mapping:
            warnings.warn("Skipping cloud base plot, missing one or more variables")
        elif cb_mapping[v1] in ds_cb.variables and cb_mapping[v2] in ds_cb.variables:
            _, _, cnt = joint_hist_contoured(
                xd=ds_cb[cb_mapping[v1]].values*xscale,
                yd=ds_cb[cb_mapping[v2]].values*yscale,
                normed_levels=normed_levels
            )

            for n, l in enumerate(cnt.collections):
                l.set_color('red')

                if n == 0:
                    l.set_label('into cloudbase')
                    lines.append(l)
                    l.set_linestyle('--')
        else:
            warnings.warn("Skipping cloud base plot, missing one or more variables")

    ax = plt.gca()
    ax.legend()

    #plt.figlegend(handles=lines, labels=[l.get_label() for l in lines], loc='right')

    # Shrink current axis by 20%
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    #ax.legend(lines=lines, loc='center left', bbox_to_anchor=(1, 0.5))

    #plt.legend()

    plt.subplots_adjust(right=0.75)
    sns.despine()

    plt.xlabel(r'{} [{}]'.format(ds_3d[v1].longname, ds_3d[v1].units))
    plt.ylabel(r'{} [{}]'.format(ds_3d[v2].longname, ds_3d[v2].units))

    if type(ds_.time.values) == float:
        plt.title("t={}hrs".format(ds_.time.values/60/60))
    else:
        plt.title("t={}".format(ds_.time.values))

    fix_axis(ax.set_xlim, v1)
    fix_axis(ax.set_ylim, v2)


def fix_axis(lim_fn, v):
    if v == 'q':
        lim_fn(14.3, 16.8)
    elif v == 't':
        lim_fn(297.6, 298.2)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(__doc__)
    


    argparser.add_argument('input_name')
    argparser.add_argument('tracking_identifier', type=str)
    argparser.add_argument('var1', type=str)
    argparser.add_argument('var2', type=str)
    argparser.add_argument('--z', type=float, nargs="+", default=Z_LEVELS_DEFAULT)
    argparser.add_argument('--mask', type=str, default=None)
    argparser.add_argument('--output-format', default='png', type=str, choices=['png', 'pdf'])
    args = argparser.parse_args()

    input_name = args.input_name
    var_name1 = args.var1
    var_name2 = args.var2
    dataset_name_with_time = input_name.split('/')[-1]
    dataset_name = input_name.split('/')[-1].split('.')[0]
    case_name = input_name.split('/')[0]

    ds_3d = get_dataset(dataset_name_with_time, variables=[var_name1, var_name2],
                        p='{}/3d_blocks/full_domain/'.format(case_name))

    if args.mask is not None:
        mask_3d = get_dataset(dataset_name_with_time,
            variables=['mask_3d.{}'.format(args.mask)],
            p='{}/masks/'.format(case_name)
        )[args.mask]
    else:
        mask_3d = None

    t0 = ds_3d.time.values

    import cloud_tracking_analysis.cloud_data

    cloud_tracking_analysis.cloud_data.ROOT_DIR = os.getcwd()
    cloud_data = CloudData(dataset_name, args.tracking_identifier,
                           dataset_pathname=case_name)


    ds_cb = get_cloudbase_data(cloud_data=cloud_data, t0=t0)

    if mask_3d is not None:
        ds_3d = ds_3d.where(mask_3d)
    main(ds_3d=ds_3d, ds_cb=ds_cb, z_levels=args.z)

    name = input_name.replace('/','__')

    title = "{} {}".format(name, plt.gca().get_title())
    out_fn = '{}.cross_correlation.{}.{}.png'.format(name, var_name1, var_name2)
    if args.mask is not None:
        title += "\nmasked by {}".format(mask_3d.longname)
        out_fn = out_fn.replace('.png', '.{}.png'.format(args.mask))

    out_fn = out_fn.replace('.png', '.{}'.format(args.output_format))

    plt.gca().set_title(title)
    plt.savefig(out_fn)
    print("Saved plot to {}".format(out_fn))
