from genesis.objects.topology.plots import shapes as plot_shapes
from genesis.objects.topology import minkowski_analytical
from genesis.utils.plot_types import multi_jointplot

import seaborn as sns
import numpy as np


def plot_reference(ax, shape, lm_range=None, linestyle='-', marker='o', 
                   x_pos=0.85, y_pos=0.6, scale=0.4, lm_diagram=2.5, **kwargs):
    try:
        fn = getattr(plot_shapes, shape)
    except AttributeError:
        raise NotImplementedError(shape)

    ds = minkowski_analytical.calc_analytical_scales(shape=shape)
    if lm_range is not None:
        ds = ds.swap_dims(dict(i='lm')).sel(lm=lm_range).swap_dims(dict(lm='i'))

    F = ds.filamentarity
    P = ds.planarity

    line, = ax.plot(P, F, linestyle=linestyle, label='spheroid', **kwargs)

    if not 'color' in kwargs:
        kwargs['color'] = line.get_color()

    for i in ds.i.values:
        ds_ = ds.sel(i=i)
        x_, y_ = ds_.planarity, ds_.filamentarity
        lm = ds_.lm.values

        lm_max = int(ds.lm.values.max())

        if int(lm) == lm or int(1.0/lm) == 1.0/lm:
            ax.plot(x_, y_, marker=marker, label='', **kwargs)
            if lm >= 1:
                s = "{:.0f}".format(lm)
                dx, dy = -4, 0
                ha = 'right'
            else:
                s = "1/{:.0f}".format(1./lm)
                dx, dy = 0, -14
                ha = 'center'
            if lm == lm_max:
                s = r"$\lambda=$"+s
            ax.annotate(s, (x_, y_), color=line.get_color(), xytext=(dx, dy),
                        textcoords='offset points', ha=ha)

    l = scale/2.0

    fn(ax, x_pos, y_pos, l=l, r=l/lm_diagram, color=line.get_color(),
       h_label=r"$\lambda r$")

    xlabel = ax.get_xlabel()
    if xlabel:
        if xlabel not in ['planarity', 'planarity [1]']:
            raise Exception(xlabel)
    else:
        ax.set_xlabel('planarity')

    ylabel = ax.get_ylabel()
    if ylabel:
        if ylabel not in ['filamentarity', 'filamentarity [1]']:
            raise Exception(ylabel)
    else:
        ax.set_ylabel('filamentarity')


def fp_plot(ds, lm_range=None):
    g = sns.jointplot('planarity', 'filamentarity', data=ds,
                      stat_func=None, marker='.')

    g.plot_joint(sns.kdeplot, cmap='Blues', zorder=0)
    ax = g.ax_joint
    ax.text(0.95, 0.95,
            "{}, {} objects".format(ds.mask, len(ds.object_id)),
            transform=ax.transAxes, horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )
    plot_reference(
        ax=ax, shape='spheroid', lm_range=lm_range, color='red', linestyle='--',
        marker='.', x_pos=0.9
    )
    ax.set_ylim(-0.01, ds.filamentarity.max())
    ax.set_xlim(-0.01, ds.planarity.max())


def main(ds, auto_scale=True):
    """
    Create a filamentarity-planarity joint plot using the `dataset` attribute
    of `ds` for the hue 
    """
    colors = ['b', 'r', 'g', 'orange']
    cmaps = ['Blues', 'Reds', 'Greens', 'Oranges']
    x = 'planarity'
    y = 'filamentarity'

    xlim = np.array([ds[x].min(), ds[x].max()])
    ylim = np.array([ds[y].min(), ds[y].max()])

    datasets = ds.dataset.values

    if len(datasets) > len(cmaps):
        raise NotImplementedError('Need to add some more colourmaps to handle'
                                  ' this number of datasets')

    g = multi_jointplot(x='planarity', y='filamentarity', z='dataset',
                    ds=ds, joint_type='kde')

    # g = sns.JointGrid(x=x, y=y, data=ds.sel(dataset=datasets[0]),)
    # for c, cmap, dataset in zip(colors, cmaps, datasets):
        # ds_ = ds.sel(dataset=dataset).dropna(dim='object_id')
        # _ = g.ax_joint.scatter(ds_[x], ds_[y], color=c, alpha=0.5, marker='.')
        # sns.kdeplot(ds_[x], ds_[y], cmap=cmap, ax=g.ax_joint, n_levels=5)
        # _ = g.ax_marg_x.hist(ds_[x], alpha=.6, color=c, range=xlim)
        # _ = g.ax_marg_y.hist(ds_[y], alpha=.6, color=c, orientation="horizontal", range=ylim)

    LABEL_FORMAT = "{name}: {count} objects"
    g.ax_joint.legend(
        labels=[LABEL_FORMAT.format(
            name=d, 
            count=int(ds.sel(dataset=d).dropna(dim='object_id').object_id.count())
            ) for d in datasets],
        bbox_to_anchor=[0.5, -0.4], loc="lower center"
    )

    if auto_scale:
        g.ax_joint.set_xlim(-0.0, 0.45)
        g.ax_joint.set_ylim(-0.0, 0.9)

    plot_reference(
        ax=g.ax_joint, shape='spheroid', color="black"
    )
