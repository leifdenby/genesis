import matplotlib
matplotlib.use("Agg")

import os

import numpy as np
import matplotlib.pyplot as plot
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager

import xarray as xr

import seaborn as sns

from tqdm import tqdm

sns.set(style='ticks', color_codes=True)
sns.despine()

VARS = [
    ("zbmn", "zcmn", "zb", "zc"),
    ("lwp_bar", "rwp_bar", ),
    ("cfrac", ),
]

N_rows = len(VARS) + 1

def main(dataset_name, dt_overview_hours=5):
    fn = os.path.join('other', '{}.ts.nc'.format(dataset_name))
    ds_ts = xr.open_dataset(fn, decode_times=False)

    assert ds_ts.time.units.startswith('seconds since 2000-01-01 0000')
    ds_ts.time.values = ds_ts.time.values/60./60.
    ds_ts.time.attrs['units'] = 'hrs'

    fig = plot.figure(figsize=(10, 9))

    font = matplotlib.font_manager.FontProperties()
    font.set_weight("bold")

    t = ds_ts.time
    t_hours = t/60./60.

    axes_ts = []

    for n, varset in enumerate(VARS):
        ax = plot.subplot2grid((N_rows,1), (n,0))
        axes_ts.append(ax)
        for var in varset:
            ax = plot.gca()
            ax.set_xlabel('time [hours]')

            da = ds_ts[var]
            da.plot(ax=ax, label=da.longname)

            if var in ('lwp_bar', 'rwp_bar'):
                ax.set_ylim(0, None)
            else:
                ax.set_ylim(0, 1.2*da[:].max())

            ax.grid(True)
            # ticks = 240.*np.arange(0, 12)
            # ax.xticks(ticks)

        if len(varset) > 1:
            ax.legend()
        else:
            ax.set_title(da.longname)

    axes_ts[0].get_shared_x_axes().join(*axes_ts)
    [ax.autoscale() for ax in axes_ts]

    # use the tick marks to determine where we'll make the overview and profile
    # plots
    t_hrs_used = filter(lambda v: v > 0.0, ax.get_xticks()[1:-1])

    fn_3d = os.path.join('raw_data', '{}.00000000.nc'.format(dataset_name))
    if os.path.exists(fn_3d):
        data_3d = xr.open_dataset(fn_3d, decode_times=False)

        x_, y_ = data_3d.time/60./60., np.zeros_like(data_3d.time)
        plot.plot(x_, y_, marker='o', linestyle='', color='red')

        for tn_, (x__, y__) in enumerate(zip(x_, y_)):
            ax.annotate(
                tn_, xy=(x__, y__),
                xytext=(0, 10), color='red',
                textcoords='offset pixels',
                horizontalalignment='center', verticalalignment='bottom'
            )

    fn = os.path.join(
        'cross_sections', 'runtime_slices',
        '{}.out.xy.lwp.nc'.format(dataset_name)
    )

    if not os.path.exists(fn):
        raise Exception("Can't find `{}`, needed for overview plots"
                        "".format(fn))
    da = xr.open_dataarray(fn, decode_times=False)

    if not da.time.units.startswith('seconds'):
        raise Exception("The `{}` has the incorrect time units (should be"
                        " seconds) which is likely because cdo mangled it."
                        " Recreate the file making sure that cdo is explicitly"
                        " told to use a *relative* time axis. The current"
                        " units are `{}`.".format(fn, da.time.units))
    
        
    print("Using times `{}` for overview plots".format(
        ", ".join([str(v) for v in t_hrs_used])
    ))

    # rescale distances to km
    def scale_dist(da, c):
        assert da[c].units == 'm'
        da[c].values = da[c].values/1000.
        da[c].attrs['units'] = 'km'
        da[c].attrs['standard_name'] = 'horz. dist.'
    scale_dist(da, 'xt')
    scale_dist(da, 'yt')

    axes_overview = []

    for n, t_ in enumerate(tqdm(t_hrs_used)):
        ax = plot.subplot2grid((N_rows, len(t_hrs_used)), (N_rows-1, n))

        da_ = da.sel(
            time=t_*60.*60., drop=True, tolerance=5.*60., method='nearest'
        ).squeeze()

        x, y = da.xt, da.yt

        (da_ > 0.001).plot.pcolormesh(
            cmap=plot.get_cmap('Greys_r'),
            rasterized=True,
            ax=ax,
            add_colorbar=False,
        )

        plot.title('t={}hrs'.format(t_))
        plot.gca().set_aspect(1)

        axes_overview.append(ax)

    plot.tight_layout()

    axes_overview[0].get_shared_y_axes().join(*axes_overview)
    [ax.autoscale() for ax in axes_overview]
    [ax.set_ylabel('') for ax in axes_overview[1:]]
    [ax.set_yticklabels([]) for ax in axes_overview[1:]]


# x = np.linspace(0, 20, 100)

# tns = [480, 960, 1440, 1920]

    
    # x, y = cloud_data.grid
    # data = cloud_data.get('lwp', tn=tn)

    # plot.pcolormesh(x/1000, y/1000, data > 0.001, cmap=plot.get_cmap('Greys_r'))
    # plot.xlabel('horizontal dist. [km]')
    # plot.xlim(-25, 25)
    # plot.ylim(-25, 25)
    # if n == 0:
        # plot.ylabel('horizontal dist. [km]')
    # plot.title('t={}min'.format(tn))
    # plot.gca().set_aspect(1)

        # for p_label, p_bounds in periods.items():
            # ax.axvspan(p_bounds[0], p_bounds[1], alpha=0.5, color='grey')
            # ylim = ax.get_ylim()
            # ax.text(.5*(p_bounds[0] + p_bounds[1]), 0.75*ylim[1], p_label, color='white', 
                    # fontsize=16, fontproperties=font, horizontalalignment='center')

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument('--filetype', default='pdf')
    argparser.add_argument('--dt_overview_hours', type=int)
    args = argparser.parse_args()

    import glob
    files = glob.glob('other/*.ts.nc')
    if len(files) == 0:
        raise Exception("Can't find *.ts.nc file, needed for this plot")
    dataset_name = os.path.basename(files[0]).split('.')[0]

    print("Plotting evolution for `{}`".format(dataset_name))

    main(dataset_name, dt_overview_hours=args.dt_overview_hours)

    plot.tight_layout()
    plot.savefig("{}.evolution.{}".format(dataset_name, args.filetype))
