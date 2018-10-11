from genesis.objects.topology.plots import shapes as plot_shapes
from genesis.objects.topology import minkowski_analytical


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
        if xlabel != 'planarity':
            raise Exception
        else:
            ax.set_xlabel('planarity')

    ylabel = ax.get_ylabel()
    if ylabel:
        if ylabel != 'filamentarity':
            raise Exception
        else:
            ax.set_ylabel('filamentarity')

