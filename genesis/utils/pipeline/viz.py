import itertools
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import luigi
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# from ...length_scales.plot import cumulant_scale_vs_height
from ...bulk_statistics import cross_correlation_with_height
from ... import length_scales
from ... import objects
from .. import plot_types, cm_nilearn

from . import data

class CumulantScalesProfile(luigi.Task):
    base_names = luigi.Parameter()
    cumulants = luigi.Parameter()
    z_max = luigi.FloatParameter(default=700.)
    plot_type = luigi.Parameter(default='scales')
    filetype = luigi.Parameter(default='pdf')

    mask = luigi.Parameter(default=None)
    mask_args = luigi.Parameter(default='')

    def _parse_cumulant_arg(self):
        cums = [c.split(':') for c in self.cumulants.split(',')]
        return [c for (n,c) in enumerate(cums) if cums.index(c) == n]

    def requires(self):
        return data.ExtractCumulantScaleProfiles(
            base_names=self.base_names,
            cumulants=self.cumulants,
            mask=self.mask,
            mask_args=self.mask_args,
        )

    def run(self):
        ds = self.input().open()

        cumulants = self._parse_cumulant_arg()
        cumulants_s = ["C({},{})".format(c[0],c[1]) for c in cumulants]

        plot_fn = length_scales.cumulant.vertical_profile.plot.plot

        import ipdb
        with ipdb.launch_ipdb_on_exception():
            plot_fn(data=ds, cumulants=cumulants_s, plot_type=self.plot_type)

        plt.savefig(self.output().path, bbox_inches='tight')

    def output(self):
        base_name = "__".join(self.base_names.split(','))
        fn = length_scales.cumulant.vertical_profile.plot.FN_FORMAT.format(
            base_name=base_name, plot_type=self.plot_type,
            mask=self.mask or "no_mask", filetype=self.filetype
        )
        return luigi.LocalTarget(fn)

class JointDistProfile(luigi.Task):
    dk = luigi.IntParameter()
    z_max = luigi.FloatParameter(significant=False, default=700.)
    v1 = luigi.Parameter()
    v2 = luigi.Parameter()
    base_name = luigi.Parameter()

    mask = luigi.Parameter(default=None)
    mask_args = luigi.Parameter(default='')
    plot_limits = luigi.ListParameter(default=None)
    data_only = luigi.BoolParameter(default=False)

    def requires(self):
        reqs = dict(
            full_domain=[
                data.ExtractField3D(field_name=self.v1, base_name=self.base_name),
                data.ExtractField3D(field_name=self.v2, base_name=self.base_name),
            ],
        )

        reqs['cloudbase'] = [
            data.ExtractCloudbaseState(base_name=self.base_name, field_name=self.v1),
            data.ExtractCloudbaseState(base_name=self.base_name, field_name=self.v2),
        ]

        if self.mask is not None:
            reqs['mask'] = data.MakeMask(method_name=self.mask,
                                         method_extra_args=self.mask_args,
                                         base_name=self.base_name
                                         )

        return reqs

    def output(self):
        if self.mask is not None:
            if not self.input()["mask"].exists():
                mask_name = 'not__a__real__mask__name'
            else:
                mask = self.input()["mask"].open()
                mask_name = mask.name
            out_fn = '{}.cross_correlation.{}.{}.masked_by.{}.png'.format(
                self.base_name, self.v1, self.v2, mask_name
            )
        else:
            out_fn = '{}.cross_correlation.{}.{}.png'.format(
                self.base_name, self.v1, self.v2
            )

        if self.data_only:
            out_fn = out_fn.replace('.png', '.nc')
            p_out = Path("data")/self.base_name/out_fn
            return data.XArrayTarget(str(p_out))
        else:
            return luigi.LocalTarget(out_fn)

    def run(self):
        ds_3d = xr.merge([
            xr.open_dataarray(r.fn) for r in self.input()["full_domain"]
        ])
        if 'cloudbase' in self.input():
            ds_cb = xr.merge([
                xr.open_dataarray(r.fn) for r in self.input()["cloudbase"]
            ])
        else:
            ds_cb = None

        if 'mask' in self.input():
            mask = self.input()["mask"].open()
            ds_3d = ds_3d.where(mask)

        ds_3d = (ds_3d.isel(zt=slice(None, None, self.dk))
                 .sel(zt=slice(0, self.z_max))
                 )

        if self.data_only:
            if 'mask' in self.input():
                ds_3d.attrs['mask_desc'] = mask.long_name
            ds_3d.to_netcdf(self.output().fn)
        else:
            ax = cross_correlation_with_height.main(ds_3d=ds_3d, ds_cb=ds_cb)

            title = ax.get_title()
            title = "{}\n{}".format(self.base_name, title)
            if 'mask' in self.input():
                title += "\nmasked by {}".format(mask.long_name)
            ax.set_title(title)

            if self.plot_limits:
                x_min, x_max, y_min, y_max = self.plot_limits

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

            plt.savefig(self.output().fn, bbox_inches='tight')


def _textwrap(s, l):
    lines = []
    line = ''

    n = 0
    in_tex = False
    for c in s:
        if c == '$':
            in_tex = not in_tex
        if n < l:
            line += c
        else:
            if c == ' ' and not in_tex:
                lines.append(line)
                line = ''
                n = 0
            else:
                line += c
        n += 1

    lines.append(line)
    return "\n".join(lines)

class JointDistProfileGrid(luigi.Task):
    dk = luigi.IntParameter()
    z_max = luigi.FloatParameter(significant=True, default=700.)
    v1 = luigi.Parameter()
    v2 = luigi.Parameter()
    base_names = luigi.Parameter()
    mask = luigi.Parameter()

    separate_axis_limits = luigi.BoolParameter(default=False)
    mask_args = luigi.Parameter(default='')

    def requires(self):
        reqs = {}

        for base_name in self.base_names.split(','):
            r = dict(
                nomask=JointDistProfile(
                    dk=self.dk, z_max=self.z_max, v1=self.v1,
                    v2=self.v2, base_name=base_name,
                    data_only=True
                ),
                masked=JointDistProfile(
                    dk=self.dk, z_max=self.z_max, v1=self.v1,
                    v2=self.v2, base_name=base_name,
                    mask=self.mask, mask_args=self.mask_args,
                    data_only=True
                ),
                cloudbase=[
                    data.ExtractCloudbaseState(base_name=base_name,
                                               field_name=self.v1),
                    data.ExtractCloudbaseState(base_name=base_name,
                                               field_name=self.v2),
                ]
            )
            reqs[base_name] = r

        return reqs

    def run(self):
        base_names = self.base_names.split(',')

        Nx, Ny = 2, len(base_names)
        if self.separate_axis_limits:
            shareaxes = 'row'
        else:
            shareaxes = True
        fig, axes = plt.subplots(nrows=len(base_names), ncols=2,
                                 sharex=shareaxes, sharey=shareaxes,
                                 figsize=(Nx*4, Ny*3+2))

        if Ny == 1:
            axes = np.array([axes,])

        for i, base_name in enumerate(base_names):
            for j, part in enumerate(['nomask', 'masked']):
                ds_3d = self.input()[base_name][part].open()
                ds_cb = xr.merge([
                    xr.open_dataarray(r.fn)
                    for r in self.input()[base_name]["cloudbase"]
                ])

                ax = axes[i,j]

                _, lines = cross_correlation_with_height.main(
                    ds_3d=ds_3d, ds_cb=ds_cb, ax=ax
                )

                if j > 0:
                    ax.set_ylabel('')
                if i < Ny-1:
                    ax.set_xlabel('')

                title = base_name
                if part == 'masked':
                    title += "\nmasked by {}".format(
                        _textwrap(ds_3d.mask_desc, 30)
                    )
                ax.set_title(title)

        lgd = plt.figlegend(
            handles=lines, labels=[l.get_label() for l in lines],
            loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.)
        )

        # rediculous hack to make sure matplotlib includes the figlegend in the
        # saved image
        ax = axes[-1,0]
        ax.text(0.5, -0.2-Ny*0.1, ' ', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(self.output().fn, bbox_inches='tight')

    def output(self):
        fn_out = "{}.cross_correlation.grid.{}.{}.png".format(
            self.base_names.replace(',', '__'), self.v1, self.v2
        )
        return luigi.LocalTarget(fn_out)

class CumulantSlices(luigi.Task):
    v1 = luigi.Parameter()
    v2 = luigi.Parameter()
    base_names = luigi.Parameter()

    z_step = luigi.IntParameter(default=4)
    z_max = luigi.FloatParameter(default=700.)
    filetype = luigi.Parameter(default='pdf')

    def requires(self):
        base_names = self.base_names.split(',')

        return dict(
            (base_name, {
                self.v1 : data.ExtractField3D(field_name=self.v1,
                                              base_name=base_name),
                self.v2 : data.ExtractField3D(field_name=self.v2,
                                              base_name=base_name),
            })
            for base_name in base_names
        )

    def makeplot(self):
        base_names = self.base_names.split(',')

        datasets = []

        for base_name in base_names:
            inputs = self.input()[base_name]
            v1 = inputs[self.v1].open(decode_times=False)
            v2 = inputs[self.v2].open(decode_times=False)

            ds = xr.merge([v1, v2])
            ds = ds.isel(zt=slice(None, None, self.z_step))
            ds = ds.sel(zt=slice(0.0, self.z_max))
            ds.attrs['name'] = base_name
            datasets.append(ds)

        plot_fn = length_scales.cumulant.sections.plot
        import ipdb
        with ipdb.launch_ipdb_on_exception():
            ax = plot_fn(datasets=datasets, var_names=[self.v1, self.v2],)

        return ax

    def run(self):
        self.makeplot()
        plt.savefig(self.output().fn, bbox_inches='tight')

    def output(self):
        fn = length_scales.cumulant.sections.FN_FORMAT_PLOT.format(
            v1=self.v1, v2=self.v2, filetype=self.filetype
        )

        return luigi.LocalTarget(fn)


class HorizontalMeanProfile(luigi.Task):
    base_name = luigi.Parameter()
    field_names = luigi.Parameter(default='qv,qc,theta')
    mask_method = luigi.Parameter(default=None)
    mask_method_extra_args = luigi.Parameter(default='')
    mask_only = luigi.BoolParameter()

    @property
    def _field_names(self):
        return self.field_names.split(',')

    def requires(self):
        reqs = dict(
            fields=[
            data.ExtractField3D(field_name=v, base_name=self.base_name)
            for v in self._field_names
            ]
        )

        if self.mask_method is not None:
            reqs['mask'] = data.MakeMask(
                    base_name=self.base_name,
                    method_name=self.mask_method,
                    method_extra_args=self.mask_method_extra_args
                )
        return reqs

    def run(self):
        ds = xr.merge([
            input.open() for input in self.input()['fields']
        ])

        fig, axes = plt.subplots(ncols=len(self._field_names), sharey=True)
        sns.set(style='ticks')

        mask = None
        if self.mask_method is not None:
            mask = self.input()['mask'].open()

        title = None
        for v, ax in zip(self._field_names, axes):
            da = ds[v]
            if mask is not None:
                if self.mask_only:
                    other = np.nan
                else:
                    other = 0.0
                da = da.where(mask, other=other)
            v_mean = da.mean(dim=('xt', 'yt'), dtype=np.float64,
                                keep_attrs=True)
            v_mean.plot(ax=ax, y='zt')
            ax.set_ylim(0, None)
            ax.set_ylabel('')
            title = ax.get_title()
            ax.set_title('')

        sns.despine()
        plt.suptitle(title)

        plt.savefig(self.output().fn)

    def output(self):
        if self.mask_method is not None:
            mask_name = data.MakeMask.make_mask_name(
                base_name=self.base_name,
                method_name=self.mask_method,
                method_extra_args=self.mask_method_extra_args
            )
            if self.mask_only:
                mask_name += "_only"
        else:
            mask_name = "nomask"

        fn = "{base_name}.{variables}.{mask_name}.mean_profile.png".format(
            base_name=self.base_name,
            variables="__".join(self._field_names),
            mask_name=mask_name,
        )
        return luigi.LocalTarget(fn)

class CrossSection(luigi.Task):
    base_names = luigi.Parameter()
    var_name = luigi.Parameter()
    z = luigi.Parameter()

    def requires(self):
        return dict([
            (base_name, data.ExtractField3D(base_name=base_name,
                                            field_name=self.var_name))
            for base_name in self.base_names.split(',')
        ])

    def run(self):
        da_ = []
        for base_name, input in self.input().items():
            da_bn = input.open()
            da_bn['base_name'] = base_name
            da_.append(da_bn)

        da = xr.concat(da_, dim='base_name')

        da.coords['xt'] /= 1000.
        da.coords['yt'] /= 1000.
        da.coords['xt'].attrs['units'] = 'km'
        da.coords['yt'].attrs['units'] = 'km'
        da.coords['xt'].attrs['long_name'] = 'horz. dist.'
        da.coords['yt'].attrs['long_name'] = 'horz. dist.'

        z = [float(v) for v in self.z.split(',')]
        da_sliced = da.sel(zt=z, method='nearest')
        da_sliced.attrs.update(da_[0].attrs)

        kws = {}
        if len(self.base_names.split(',')) > 1:
            kws['col'] = 'base_name'
        if len(z) > 1:
            kws['row'] = 'zt'
        if self.var_name.startswith('d_'):
            kws['center'] = 0.0
        da_sliced.plot(rasterized=True, robust=True, **kws)

        plt.savefig(self.output().fn, bbox_inches='tight')

    def output(self):
        fn = "{}.{}.png".format(self.base_names.replace(',', '__'), self.var_name)
        return luigi.LocalTarget(fn)

class ObjectScalesComparison(luigi.Task):
    plot_definition = luigi.Parameter()
    not_pairgrid = luigi.BoolParameter(default=False)
    file_type = luigi.Parameter(default='png')

    def _parse_plot_definition(self):
        try:
            with open('{}.yaml'.format(self.plot_definition)) as fh:
                loader = getattr(yaml, 'FullLoader', yaml.Loader)
                return yaml.load(fh, Loader=loader)
        except IOError as e:
            raise e

    def requires(self):
        plot_definition = self._parse_plot_definition()

        def _make_dataset_label(**kws):
            return ", ".join([
                "{}={}".format(str(k), str(v))
                for (k,v) in kws.items()
            ])

        global_kws = plot_definition['global']

        variables = set(global_kws.pop('variables').split(','))

        return dict([
            (
                _make_dataset_label(**kws),
                data.ComputeObjectScales(variables=",".join(variables),
                                         **global_kws, **kws)
            )
            for kws in plot_definition['sources']
        ])

    def _parse_filters(self, filter_defs):
        if filter_defs is None:
            return []

        ops = {
            '>': lambda f, v: f > v,
            '<': lambda f, v: f < v,
        }
        filters = []

        for filter_s in filter_defs.split(','):
            found_op = False
            for op_str, func in ops.items():
                if op_str in filter_s:
                    field, value = filter_s.split(op_str)
                    filters.append((field, func, value))
                    found_op = True

            if not found_op:
                raise NotImplementedError(filter_s)

        return filters

    def _apply_filters(self, ds, filter_defs):
        for field, func, value in self._parse_filters(filter_defs):
            ds = ds.where(func(ds[field], float(value)))

        return ds

    def _load_data(self):
        def _add_dataset_label(label, input):
            ds = input.open(decode_times=False)
            ds['dataset'] = label
            return ds

        ds = xr.concat(
            [
                _add_dataset_label(k, input)
                for (k, input) in self.input().items()
            ],
            dim='dataset'
        )
        plot_definition = self._parse_plot_definition()
        global_params = plot_definition['global']

        ds = self._apply_filters(
            ds=ds,
            filter_defs=global_params.get('filters', None)
        )

        if ds.object_id.count() == 0:
            raise Exception("After filter operations there is nothing to plot!")

        return ds

    def _set_suptitle(self):
        plot_definition = self._parse_plot_definition()
        global_params = plot_definition['global']
        global_params.pop('variables')

        identifier = "\n".join([
            "{}={}".format(str(k), str(v))
            for (k,v) in global_params.items()
        ])

        plt.suptitle(identifier, y=[1.1, 1.5][self.not_pairgrid])

    def run(self):
        ds = self._load_data()

        plot_definition = self._parse_plot_definition()
        global_params = plot_definition['global']
        variables = global_params.pop('variables').split(',')
        objects.topology.plots.overview(
            ds=ds, as_pairgrid=not self.not_pairgrid, variables=variables
        )

        self._set_suptitle()

        plt.savefig(self.output().fn, bbox_inches='tight')

    def output(self):
        fn = "{}.object_scales.{}".format(
            self.plot_definition, self.file_type
        )
        return luigi.LocalTarget(fn)


class FilamentarityPlanarityComparison(ObjectScalesComparison):
    def run(self):
        ds = self._load_data()
        objects.topology.plots.filamentarity_planarity(ds=ds)
        self._set_suptitle()
        plt.savefig(self.output().fn, bbox_inches='tight')


    def output(self):
        fn_base = super().output().fn

        return luigi.LocalTarget(
            fn_base.replace('.object_scales.', '.filamentarity_planarity.')
        )

class ObjectScalesFit(luigi.Task):
    var_name = luigi.Parameter(default='length')
    dv = luigi.FloatParameter(default=None)
    v_max = luigi.FloatParameter(default=None)
    file_type = luigi.Parameter(default='png')
    plot_components = luigi.Parameter(default='default')
    plot_size = luigi.Parameter(default='3,3')

    base_names = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')
    object_splitting_scalar = luigi.Parameter()
    object_filters = luigi.Parameter(default=None)

    def requires(self):
        return dict([
            (
                base_name,
                data.ComputeObjectScales(
                    variables=self.var_name, base_name=base_name,
                    mask_method=self.mask_method,
                    mask_method_extra_args=self.mask_method_extra_args,
                    object_splitting_scalar=self.object_splitting_scalar,
                    object_filters=self.object_filters,
                    )
            )
            for base_name in self.base_names.split(',')
        ])

    def get_base_name_labels(self):
        return {}

    def get_suptitle(self):
        s_filters = objects.property_filters.latex_format(self.object_filters)
        return "{}\n{}".format(self.base_names,s_filters)

    def run(self):
        inputs = self.input()
        if self.plot_components == 'default':
            plot_components = 'default'
            Nc = 4
        else:
            plot_components = self.plot_components.split(',')
            Nc = len(plot_components)

        sx, sy = [float(v) for v in self.plot_size.split(',')]
        fig, axes = plt.subplots(
            ncols=Nc, nrows=len(inputs), figsize=(sx*Nc, sy*len(inputs)), 
            sharex="col", sharey='col'
        )

        if len(axes.shape) == 1:
            axes = np.array([axes,])

        for n, (base_name, input) in enumerate(inputs.items()):
            input = input.open()
            if isinstance(input, xr.Dataset):
                ds = input
                da_v = ds[self.var_name]
            else:
                da_v = input
            if self.dv is not None:
                da_v = da_v[da_v > self.dv]
            else:
                da_v = da_v[da_v > 0.0]

            plot_to = axes[n]
            length_scales.model_fitting.exponential_fit.fit(
                da_v, dv=self.dv, debug=False, plot_to=plot_to,
                plot_components=plot_components
            )

            ax = plot_to[0]
            desc = self.get_base_name_labels().get(base_name)
            if desc is None:
                desc = base_name.replace('_', ' ').replace('.', ' ')
            desc += "\n({} objects)".format(len(da_v.object_id))
            ax.text(-0.5, 0.5, desc, transform=ax.transAxes,
                    horizontalalignment='right')

        sns.despine()

        if self.v_max:
            [ax.set_xlim(0, self.v_max) for ax in axes[:,:2].flatten()]
        if da_v.units == 'm':
            [ax.set_ylim(1.0e-6, None) for ax in axes[:,1].flatten()]
        if axes.shape[0] > 1:
            [ax.set_xlabel('') for ax in axes[0].flatten()]
        [ax.set_title('') for ax in axes[1:].flatten()]
        plt.suptitle(self.get_suptitle(), y=1.1)
        plt.tight_layout()
        plt.savefig(self.output().fn, bbox_inches='tight')

    def output(self):
        s_filter = ''
        if self.object_filters is not None:
            s_filter = '.filtered_by.{}'.format(
                (self.object_filters.replace(',','.')
                                    .replace(':', '__')
                                    .replace('=', '_')
                )
            )
        fn = "object_scales_exp_fit.{}.{}{}.{}".format(
            self.var_name,
            self.base_names.replace(',', '__'),
            s_filter,
            self.file_type
        )
        return luigi.LocalTarget(fn)

class ObjectsScaleDist(luigi.Task):
    var_name = luigi.Parameter()
    dv = luigi.FloatParameter(default=None)
    v_max = luigi.FloatParameter(default=None)
    file_type = luigi.Parameter(default='png')
    show_cumsum = luigi.BoolParameter(default=False)
    cumsum_markers = luigi.Parameter(default=None)
    as_density = luigi.Parameter(default=False)
    figsize = luigi.Parameter(default='6,6')

    base_names = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')
    object_splitting_scalar = luigi.Parameter()
    object_filters = luigi.Parameter(default=None)

    def requires(self):
        reqs = {}
        for var_name in self.var_name.split(','):
            reqs[var_name] = dict([
                (
                    base_name,
                    data.ComputeObjectScales(
                        variables=self.var_name, base_name=base_name,
                        mask_method=self.mask_method,
                        mask_method_extra_args=self.mask_method_extra_args,
                        object_splitting_scalar=self.object_splitting_scalar,
                        object_filters=self.object_filters,
                        )
                )
                for base_name in self.base_names.split(',')
            ])
        return reqs

    @staticmethod
    def _calc_fixed_bin_args(v, dv):
        vmin = np.floor(v.min()/dv)*dv
        vmax = np.ceil(v.max()/dv)*dv
        nbins = int((vmax-vmin)/dv)
        return dict(range=(vmin, vmax), bins=nbins)

    def get_base_name_labels(self):
        return {}

    def get_title(self):
        return ''

    def run(self):
        figsize = [float(v) for v in self.figsize.split(',')]
        N_vars = len(self.var_name.split(','))
        N_basenames = len(self.base_names.split(','))
        fig, axes = plt.subplots(figsize=(figsize[0]*N_vars, figsize[0]), ncols=N_vars,
                                 sharey=True)
        if N_vars == 1:
            axes = [axes,]

        d_units = []
        for n, (var_name, inputs) in enumerate(self.input().items()):
            ax = axes[n]

            if self.show_cumsum:
                ax_twin = ax.twinx()

            bins = None
            for n, (base_name, input) in enumerate(inputs.items()):
                input = input.open()
                if isinstance(input, xr.Dataset):
                    ds = input
                    da_v = ds[var_name]
                else:
                    da_v = input

                da_v = da_v[np.logical_and(~np.isnan(da_v),~np.isinf(da_v))]
                desc = self.get_base_name_labels().get(base_name)
                if desc is None:
                    desc = base_name.replace('_', ' ').replace('.', ' ')
                desc += " ({} objects)".format(int(da_v.object_id.count()))
                kws = dict(density=self.as_density)
                if self.dv is not None:
                    kws.update(self._calc_fixed_bin_args(v=da_v.values, dv=self.dv))
                if bins is not None:
                    kws['bins'] = bins
                _, bins, pl_hist = da_v.plot.hist(ax=ax, alpha=0.4, label=desc, **kws)

                if self.show_cumsum:
                    # cumulative dist. plot
                    x_ = np.sort(da_v)
                    y_ = np.cumsum(x_)
                    c = pl_hist[0].get_facecolor()
                    ax_twin.plot(x_, y_, color=c, marker='.', linestyle='', markeredgecolor="None")
                    ax_twin.axhline(y=y_[-1], color=c, linestyle='--', alpha=0.3)

                    if self.cumsum_markers is not None:
                        markers = [float(v) for v in self.cumsum_markers.split(',')]
                        for m in markers:
                            i = np.nanargmin(np.abs(m*y_[-1]-y_))
                            x_m = x_[i]
                            ax_twin.axvline(x_m, color=c, linestyle=':')

            ax.set_title(self.get_title())
            if self.as_density:
                ax.set_ylabel('object density [1/{}]'.format(da_v.units))
            else:
                ax.set_ylabel('num objects')
            if self.show_cumsum:
                ax_twin.set_ylabel('sum of {}'.format(xr.plot.utils.label_from_attrs(da_v)))

            d_units.append(da_v.units)

        if all([d_units[0] == u for u in d_units[1:]]):
            ax1 = axes[0]
            [ax1.get_shared_x_axes().join(ax1, ax) for ax in axes[1:]]
            ax1.autoscale()
            [ax.set_ylabel('') for ax in axes[1:]]

        s_filters = objects.property_filters.latex_format(self.object_filters)
        st = plt.suptitle("{}\n{}".format(self.base_names,s_filters), y=1.1)

        sns.despine(right=not self.show_cumsum)
        ax_lgd = axes[len(axes)//2]
        lgd = ax_lgd.legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.15-0.1*N_basenames),
        )

        if self.v_max is not None:
            ax.set_xlim(0., self.v_max)

        plt.tight_layout()
        plt.savefig(self.output().fn, bbox_inches='tight', bbox_extra_artists=(lgd, st))

    def output(self):
        s_filter = ''
        if self.object_filters is not None:
            s_filter = '.filtered_by.{}'.format(
                (self.object_filters.replace(',','.')
                                    .replace(':', '__')
                                    .replace('=', '_')
                )
            )
        fn = "objects_scale_dist.{}.{}{}.{}".format(
            self.var_name.replace(',', '__'),
            self.base_names.replace(',', '__'),
            s_filter,
            self.file_type
        )
        return luigi.LocalTarget(fn)

class ObjectsScalesJointDist(luigi.Task):
    x = luigi.Parameter()
    y = luigi.Parameter()
    file_type = luigi.Parameter(default='png')

    xmax = luigi.FloatParameter(default=None)
    ymax = luigi.FloatParameter(default=None)
    plot_type = luigi.Parameter(default='scatter')
    plot_aspect = luigi.FloatParameter(default=None)
    plot_annotations = luigi.Parameter(default=None)
    scaling = luigi.Parameter(default=None)

    base_names = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')
    object_splitting_scalar = luigi.Parameter()
    object_filters = luigi.Parameter(default=None)

    def requires(self):
        return dict([
            (
                base_name,
                data.ComputeObjectScales(
                    variables="{},{}".format(self.x, self.y),
                    base_name=base_name,
                    mask_method=self.mask_method,
                    mask_method_extra_args=self.mask_method_extra_args,
                    object_splitting_scalar=self.object_splitting_scalar,
                    object_filters=self.object_filters,
                    )
            )
            for base_name in self.base_names.split(',')
        ])

    @staticmethod
    def _calc_fixed_bin_args(v, dv):
        vmin = np.floor(v.min()/dv)*dv
        vmax = np.ceil(v.max()/dv)*dv
        nbins = int((vmax-vmin)/dv)
        return dict(range=(vmin, vmax), bins=nbins)

    def make_plot(self):
        inputs = self.input()

        kws = {}
        if self.xmax is not None:
            kws['xlim'] = (0, self.xmax)
        if self.ymax is not None:
            kws['ylim'] = (0, self.ymax)

        if self.plot_type.startswith('jointplot'):
            if '_' in self.plot_type:
                kws['joint_type'] = self.plot_type.split('_')[-1]

            def _lab(label, ds_):
                ds_['dataset'] = label
                return ds_
            dss = [
                _lab(name, input.open()) for (name, input) in inputs.items()
            ]
            ds = xr.concat(dss, dim='dataset')

            g = plot_types.multi_jointplot(x=self.x, y=self.y, z='dataset', ds=ds, **kws)
            ax = g.ax_joint
            s_filters = objects.property_filters.latex_format(self.object_filters)
            plt.suptitle("{}\n{}".format(self.base_names,s_filters), y=1.1)
            if self.plot_aspect is not None:
                raise Exception("Can't set aspect ratio on jointplot, set limits instead")
        elif self.plot_type in ['scatter', 'scatter_hist']:
            ax = None
            alpha = 1.0/len(inputs)
            if ax is None:
                fig, ax = plt.subplots()
            for n, (base_name, input) in enumerate(inputs.items()):
                print(base_name)
                ds = input.open()
                da_v1 = ds[self.x]
                da_v2 = ds[self.y]

                desc = base_name.replace('_', ' ').replace('.', ' ')
                desc += " ({} objects)".format(len(da_v1))
                if self.plot_type == 'scatter':
                    ax.scatter(x=da_v1.values, y=da_v2.values, alpha=alpha,
                               label=desc, s=5.)
                else:
                    ax = plot_types.make_marker_plot(x=da_v1.values,
                                                     y=da_v2.values,
                                                     alpha=alpha)

            ax.set_xlabel(xr.plot.utils.label_from_attrs(da_v1))
            ax.set_ylabel(xr.plot.utils.label_from_attrs(da_v2))
            sns.despine()
            ax.legend()
            if self.plot_aspect is not None:
                ax.set_aspect(self.plot_aspect)

            plt.title("{}\n{}".format(self.base_names, self.object_filters))
            if self.xmax is not None:
                ax.set_xlim(np.nanmin(da_v1), self.xmax)
            else:
                xmax = np.nanmax(da_v1)
                xmin = np.nanmin(da_v1)
                ax.set_xlim(xmin, xmax)
            if self.ymax is not None:
                ymin = [0.0, None][np.nanmin(da_v2) < 0.0]
                ax.set_ylim(ymin, self.ymax)
        else:
            raise NotImplementedError(self.plot_type)

        # for n, (base_name, input) in enumerate(inputs.items()):
            # ds = input.open()
            # da_v1 = ds[self.x]
            # da_v2 = ds[self.y]

            # desc = base_name.replace('_', ' ').replace('.', ' ')
            # if hue_label:
                # ds_ = ds.where(ds[hue_label], drop=True)
                # ds_[v].plot.hist(ax=ax, bins=bins)
            # g = sns.jointplot(x=self.x, y=self.y, , s=10)

            # ax = g.ax_joint
            # ax.set_xlabel(xr.plot.utils.label_from_attrs(da_v1))
            # ax.set_ylabel(xr.plot.utils.label_from_attrs(da_v2))

        if self.plot_annotations is not None:
            for annotation in self.plot_annotations.split(','):
                if (annotation == "plume_vs_thermal"
                    and self.x == 'z_proj_length'
                    and self.y == "z_min"):
                    z_b = 200.
                    z_cb = 600.
                    if self.xmax is None:
                        xmax = ax.get_xlim()[1]
                    else:
                        xmax = self.xmax
                    x_ = np.linspace(z_cb-z_b, xmax, 100)

                    ax.plot(x_, z_cb-x_, marker='', color='grey',
                            linestyle='--', alpha=0.6)
                    t_kws = dict(transform=ax.transData,
                            color='grey', horizontalalignment='center')
                    ax.text(356., 100., "thermals", **t_kws)
                    ax.text(650., 100., "plumes", **t_kws)
                elif annotation == "unit_line":
                    x_ = np.linspace(
                        max(ax.get_xlim()[0], ax.get_ylim()[0]),
                        min(ax.get_xlim()[-1], ax.get_ylim()[-1]),
                        100
                    )
                    ax.plot(x_, x_, linestyle='--', alpha=0.6, color='grey')
                else:
                    raise NotImplementedError(annotation, self.x, self.y)

        if self.scaling is None:
            pass
        elif self.scaling == "loglog":
            ax.set_yscale('log')
            ax.set_xscale('log')
        else:
            raise NotImplementedError(self.scaling)

        plt.tight_layout()

        return ax

    def run(self):
        ax = self.make_plot()

        try:
            plt.savefig(self.output().fn, bbox_inches='tight')
        except IOError:
            import hues
            import uuid
            fn = "plot.{}.{}".format(str(uuid.uuid4()), self.file_type)
            hues.warn("filename became to long, saved to `{}`".format(fn))
            plt.savefig(fn, bbox_inches='tight')

    def output(self):
        s_filter = ''
        if self.object_filters is not None:
            s_filter = '.filtered_by.{}'.format(
                (self.object_filters.replace(',','.')
                                    .replace(':', '__')
                                    .replace('=', '_')
                )
            )

        objects_name = data.IdentifyObjects.make_name(
            base_name=self.base_names,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            filter_defs=self.object_filters,
        )

        fn = "objects_scales_joint_dist.{}__{}.{}{}.{}".format(
            self.x,
            self.y,
            objects_name.replace(',', '__'),
            s_filter,
            self.file_type
        )
        return luigi.LocalTarget(fn)

class ObjectScaleVsHeightComposition(luigi.Task):
    x = luigi.Parameter()
    field_name = luigi.Parameter()

    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')
    object_splitting_scalar = luigi.Parameter()

    object_filters = luigi.Parameter(default=None)
    dx = luigi.FloatParameter(default=None)
    z_max = luigi.FloatParameter(default=None)
    filetype = luigi.Parameter(default='png')
    x_max = luigi.FloatParameter(default=None)

    def requires(self):
        return dict(
            decomp_profile=data.ComputePerObjectProfiles(
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                object_splitting_scalar=self.object_splitting_scalar,
                field_name=self.field_name,
                op='sum',
                z_max=self.z_max,
            ),
            da_3d=data.ExtractField3D(
                base_name=self.base_name,
                field_name=self.field_name,
            ),
            scales=data.ComputeObjectScales(
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                object_splitting_scalar=self.object_splitting_scalar,
                variables=self.x,
                object_filters=self.object_filters,
            ),
            mask=data.MakeMask(
                base_name=self.base_name,
                method_name=self.mask_method,
                method_extra_args=self.mask_method_extra_args,
            ),
        )

    def run(self):
        input = self.input()
        da_field = input['decomp_profile'].open()
        da_3d = input['da_3d'].open()
        ds_scales = input['scales'].open()
        da_mask = input['mask'].open()
        nx, ny = da_3d.xt.count(), da_3d.yt.count()

        da_prof_ref = da_3d.where(da_mask).sum(dim=('xt', 'yt'),
                                               dtype=np.float64)/(nx*ny)

        if self.object_filters is not None:
            da_field.where(da_field.object_id == ds_scales.object_id)

        ds = xr.merge([da_field, ds_scales])

        if self.z_max is not None:
            ds = ds.sel(zt=slice(None, self.z_max))

        ds = ds.where(np.logical_and(
            ~np.isinf(ds[self.x]),
            ~np.isnan(ds[self.x]),
        ), drop=True)

        ax = objects.flux_contribution.plot(
            ds=ds, x=self.x, v=self.field_name + '__sum',
            nx=nx, ny=ny, dx=self.dx, da_prof_ref=da_prof_ref,
        )

        if self.x_max is not None:
            ax.set_xlim(None, self.x_max)

        N_objects = int(ds.object_id.count())
        plt.suptitle(self.get_suptitle(N_objects=N_objects), y=1.0)

        plt.savefig(self.output().fn, bbox_inches='tight')

    def get_suptitle(self, N_objects):
        s_filters = objects.property_filters.latex_format(self.object_filters)
        return "{} ({} objects)\n{}".format(self.base_name,N_objects, s_filters)


    def output(self):
        mask_name = data.MakeMask.make_mask_name(
            base_name=self.base_name,
            method_name=self.mask_method,
            method_extra_args=self.mask_method_extra_args
        )
        s_filter = ''
        if self.object_filters is not None:
            s_filter = '.filtered_by.{}'.format(
                (self.object_filters.replace(',','.')
                                    .replace(':', '__')
                                    .replace('=', '_')
                )
            )
        fn = ("{base_name}.{mask_name}.{field_name}__by__{x}"
             "{s_filter}.{filetype}".format(
            base_name=self.base_name, mask_name=mask_name,
            field_name=self.field_name, x=self.x, filetype=self.filetype,
            s_filter=s_filter,
        ))
        target = luigi.LocalTarget(fn)
        return target
