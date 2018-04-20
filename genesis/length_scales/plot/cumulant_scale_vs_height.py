"""
Routines for plotting cumulant characteristics from netCDF datafile
"""
import matplotlib
matplotlib.use("Agg")

import warnings
import os
import re

import xarray as xr
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plot


FULL_SUITE_PLOT_PARTS = dict(
    l=(0,0),
    q=(0,slice(1,None)),
    t=(1,slice(0,None)),
    w=(2,slice(0,2)),
    q_flux=(2,2),
    t_flux=(2,3),
    l_flux=(2,4),
)


def fix_cumulant_name(name):
    name_mapping = {
        'q': 'q_t',
        't': r"\theta_l",
        'l': 'q_l',
        'q_flux': r"\overline{w'q_t'}",
        't_flux': r"\overline{w'\theta_l}",
        'l_flux': r"\overline{w'q_l'}",
    }

    v1, v2, extra = re.match('C\((\w+),(\w+)\)(.*)', name).groups()

    v1_latex = name_mapping.get(v1, v1)
    v2_latex = name_mapping.get(v2, v2)

    return r"$C({},{})$".format(v1_latex, v2_latex) + '\n' + extra


def plot_full_suite(data, marker=''):
    gs = GridSpec(3, 5)
    figure = plot.figure(figsize=(10, 12))

    aspect = 0.3


    ax = None
    for var_name, s in FULL_SUITE_PLOT_PARTS.items():
        ax = plot.subplot(gs[s], sharey=ax, adjustable='box-forced')

        d_ = data.sel(cumulant='C(l,l)', drop=True)
        z_cb = d_.where(d_.width_principle>0.1, drop=True).zt.min()
        ax.axhline(z_cb, linestyle=':', color='grey', alpha=0.6)

        for p in data.dataset_name.values:
            lines = []

            cumulant = "C({},{})".format(var_name, var_name)
            d = data.sel(dataset_name=p, drop=True).sel(cumulant=cumulant, drop=True)

            line, = plot.plot(d.width_principle, d.zt, marker=marker,
                              label="{} principle".format(str(p)))
            line2, = plot.plot(d.width_perpendicular, d.zt, marker=marker,
                               label="{} orthog.".format(str(p)),
                               linestyle='--', color=line.get_color())

            plot.title(fix_cumulant_name(cumulant))

            lines.append(line)
            lines.append(line2)

            plot.xlabel("characterisc width [m]")
        
        if s[1] == 0 or type(s[1]) == slice and s[1].start == 0:
            plot.ylabel('height [m]')
        else:
            plot.setp(ax.get_yticklabels(), visible=False)

    plot.tight_layout()

    for ax in figure.axes:
        # once the plots have been rendered we want to resize the xaxis so 
        # that the aspect ratio is the same, note sharey on the subplot above
        _, _, w, h = ax.get_position().bounds
        ax.set_xlim(0, d.zt.max()/h*w*aspect)    

    plot.subplots_adjust(bottom=0.10)
    lgd = plot.figlegend(lines, [l.get_label() for l in lines], loc='lower center', ncol=2)


def plot_default(data, marker='', z_max=None, cumulants=[], split_subplots=True,
                 with_legend=True, fig=None):

    if len(cumulants) == 0:
        cumulants = data.cumulant.values

    if z_max is not None:
        data = data.copy().where(data.zt < z_max, drop=True)

    if fig is None and split_subplots:
        fig = plot.figure(figsize=(2.5*len(cumulants), 4))

    z_ = data.zt

    ax = None

    for i, cumulant in enumerate(cumulants):
        lines = []
        n = data.cumulant.values.tolist().index(cumulant)
        s = data.isel(cumulant=n, drop=True).squeeze()
        if split_subplots:
            ax = plot.subplot(1,len(cumulants),i+1, sharey=ax)
        else:
            ax = plot.gca()
        for p in data.dataset_name.values:
            d = data.sel(dataset_name=p, drop=True).sel(cumulant=cumulant, drop=True)

            line, = plot.plot(d.width_principle, d.zt, marker=marker,
                              label="{} principle".format(str(p)))
            line2, = plot.plot(d.width_perpendicular, d.zt, marker=marker,
                               label="{} orthog.".format(str(p)),
                               linestyle='--', color=line.get_color())

            lines.append(line)
            lines.append(line2)

        plot.title(fix_cumulant_name(cumulant))
        plot.tight_layout()
        plot.xlabel("characterisc width [m]")
        
        if i == 0:
            plot.ylabel('height [m]')
        else:
            plot.setp(ax.get_yticklabels(), visible=False)

    if with_legend:
        plot.subplots_adjust(bottom=0.24)
        lgd = plot.figlegend(lines, [l.get_label() for l in lines], loc='lower center', ncol=2)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plot

    import seaborn as sns
    sns.set(style='ticks')

    import argparse
    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument('input', help='input netCDF file')
    argparser.add_argument('--vars', help='variables for cumulants', nargs='*',
        default=[], type=str,)
    argparser.add_argument('--z_max', help='max height', default=None,
        type=float,)

    args = argparser.parse_args()

    dataset = xr.open_dataset(args.input)

    do_full_suite_plot = None
    variables = args.vars
    if len(variables) == 0:
        variables = FULL_SUITE_PLOT_PARTS.keys()
        do_full_suite_plot = True
    else:
        do_full_suite_plot = False


    cumulants = [
        "C({},{})".format(v,v) for v in variables
    ]

    missing_cumulants = [
        c for c in cumulants if not c in dataset.cumulant.values
    ]

    if do_full_suite_plot:
        if not len(missing_cumulants) == 0:
            warnings.warn("Not all variables for full suite plot, missing: {}"
                          "".format(", ".join(missing_cumulants)))

            plot_default(dataset, z_max=args.z_max, cumulants=cumulants)
        else:
            plot_full_suite(dataset)
    else:
        if not len(missing_cumulants) == 0:
            raise Exception("Not all variables for plot, missing: {}"
                          "".format(", ".join(missing_cumulants)))

        else:
            plot_default(dataset, z_max=args.z_max, cumulants=cumulants)

    sns.despine()

    fn = os.path.basename(__file__).replace('.pyc', '.pdf')\
                                   .replace('.py', '.pdf')
    plot.savefig(fn, bbox_inches='tight')

    print("Saved figure to {}".format(fn))
