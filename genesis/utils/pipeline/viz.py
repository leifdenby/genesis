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

from . import data

class ExtractCumulantScaleProfile(luigi.Task):
    base_name = luigi.Parameter()
    v1 = luigi.Parameter()
    v2 = luigi.Parameter()
    z_max = luigi.FloatParameter(default=700.)
    mask = luigi.Parameter(default=None)
    mask_args = luigi.Parameter(default='')
    width_method = length_scales.cumulant.calc.WidthEstimationMethod.MASS_WEIGHTED

    def requires(self):
        reqs = {}
        reqs['fields'] = [
                data.ExtractField3D(base_name=self.base_name,
                                    field_name=self.v1),
                data.ExtractField3D(base_name=self.base_name,
                                    field_name=self.v2),
        ]

        if self.mask is not None:
            reqs['mask'] = data.MakeMask(method_name=self.mask,
                                         method_extra_args=self.mask_args,
                                         base_name=self.base_name
                                         )

        return reqs

    def run(self):
        da_v1 = self.input()['fields'][0].open(decode_times=False)
        da_v2 = self.input()['fields'][1].open(decode_times=False)

        calc_fn = length_scales.cumulant.vertical_profile.calc.get_height_variation_of_characteristic_scales

        mask = None
        if self.mask:
            mask = self.input()['mask'].open(decode_times=False)


        import ipdb
        with ipdb.launch_ipdb_on_exception():
            da = calc_fn(
                v1_3d=da_v1, v2_3d=da_v2, width_method=self.width_method,
                z_max=self.z_max, mask=mask
            )

        da.to_netcdf(self.output().path)

    def output(self):
        fn = length_scales.cumulant.vertical_profile.calc.FN_FORMAT.format(
            base_name=self.base_name, v1=self.v1, v2=self.v2,
            mask=self.mask or "no_mask"
        )
        return data.XArrayTarget(fn)

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
        reqs = {}

        for base_name in self.base_names.split(','):
            reqs[base_name] = [
                ExtractCumulantScaleProfile(
                    base_name=base_name, v1=c[0], v2=c[1],
                    mask=self.mask, mask_args=self.mask_args,
                )
                for c in self._parse_cumulant_arg()
            ]

        return reqs

    def run(self):
        datasets = []
        for base_name in self.base_names.split(','):
            ds_ = xr.concat([
                input.open(decode_times=False)
                for input in self.input()[base_name]
            ], dim='cumulant')
            ds_['dataset_name'] = base_name
            datasets.append(ds_)

        ds = xr.concat(datasets, dim='dataset_name')

        cumulants = self._parse_cumulant_arg()
        cumulants_s = ["C({},{})".format(c[0],c[1]) for c in cumulants]

        plot_fn = length_scales.cumulant.vertical_profile.plot.plot

        import ipdb
        with ipdb.launch_ipdb_on_exception():
            plot_fn(data=ds, cumulants=cumulants_s, plot_type=self.plot_type)

        print(self.output().path)

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

    def run(self):
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

            print(v1.name, v1.xt)
            print(v2.name, v2.xt)

        plot_fn = length_scales.cumulant.sections.plot
        import ipdb
        with ipdb.launch_ipdb_on_exception():
            plot_fn(datasets=datasets, var_names=[self.v1, self.v2],)

        plt.savefig(self.output().fn, bbox_inches='tight')

    def output(self):
        fn = length_scales.cumulant.sections.FN_FORMAT_PLOT.format(
            v1=self.v1, v2=self.v2, filetype=self.filetype
        )

        return luigi.LocalTarget(fn)


class HorizontalMeanProfile(luigi.Task):
    base_name = luigi.Parameter()
    field_names = luigi.Parameter(default='qv,qc,theta')

    @property
    def _field_names(self):
        return self.field_names.split(',')

    def requires(self):
        return [
            data.ExtractField3D(field_name=v, base_name=self.base_name)
            for v in self._field_names
        ]

    def run(self):
        ds = xr.merge([
            input.open() for input in self.input()
        ])

        fig, axes = plt.subplots(ncols=len(self._field_names), sharey=True)
        sns.set(style='ticks')

        title = None
        for v, ax in zip(self._field_names, axes):
            v_mean = ds[v].mean(dim=('xt', 'yt'), dtype=np.float64,
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
        fn = "{base_name}.{variables}.mean_profile.pdf".format(
            base_name=self.base_name,
            variables="__".join(self._field_names)
        )
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


class MinkowskiCharacteristicScalesFit(luigi.Task):
    var_name = luigi.Parameter(default='length')
    dv = luigi.FloatParameter(default=25.)
    v_max = luigi.FloatParameter(default=400.)
    file_type = luigi.Parameter(default='png')

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

    def run(self):
        inputs = self.input()
        fig, axes = plt.subplots(
            ncols=4, nrows=len(inputs), figsize=(14, 3*len(inputs)), 
            sharex="col", sharey='col'
        )

        for n, (base_name, input) in enumerate(inputs.items()):
            input = input.open()
            if isinstance(input, xr.Dataset):
                ds = input
                da_v = ds[self.var_name]
            else:
                da_v = input
            da_v = da_v[da_v > 25.]
            plot_to = axes[n]
            length_scales.minkowski.exponential_fit.fit(
                da_v, dv=25., debug=False, plot_to=plot_to
            )

            ax = plot_to[0]
            desc = base_name.replace('_', ' ').replace('.', ' ')
            ax.text(-0.3, 0.5, desc, transform=ax.transAxes,
                    horizontalalignment='right')

        sns.despine()

        [ax.set_xlim(0, self.v_max) for ax in axes[:,:2].flatten()]
        [ax.set_ylim(1.0e-6, None) for ax in axes[:,1].flatten()]
        [ax.set_xlabel('') for ax in axes[0].flatten()]
        [ax.set_title('') for ax in axes[1:].flatten()]
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
        fn = "minkowski_scales_exp_fit.{}.{}{}.{}".format(
            self.var_name,
            self.base_names.replace(',', '__'),
            s_filter,
            self.file_type
        )
        return luigi.LocalTarget(fn)

class ObjectsScaleDist(luigi.Task):
    var_name = luigi.Parameter(default='length')
    dv = luigi.FloatParameter(default=25.)
    v_max = luigi.FloatParameter(default=400.)
    file_type = luigi.Parameter(default='png')

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

    @staticmethod
    def _calc_fixed_bin_args(v, dv):
        vmin = np.floor(v.min()/dv)*dv
        vmax = np.ceil(v.max()/dv)*dv
        nbins = int((vmax-vmin)/dv)
        return dict(range=(vmin, vmax), bins=nbins)

    def run(self):
        inputs = self.input()
        fig, ax = plt.subplots()

        for n, (base_name, input) in enumerate(inputs.items()):
            input = input.open()
            if isinstance(input, xr.Dataset):
                ds = input
                da_v = ds[self.var_name]
            else:
                da_v = input

            desc = base_name.replace('_', ' ').replace('.', ' ')
            da_v.plot.hist(ax=ax, alpha=0.4, label=desc,
                           **self._calc_fixed_bin_args(
                               v=da_v.values, dv=self.dv)
                          )

        sns.despine()
        plt.legend()
        if self.v_max is not None:
            ax.set_xlim(0., self.v_max)

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
        fn = "objects_scale_dist.{}.{}{}.{}".format(
            self.var_name,
            self.base_names.replace(',', '__'),
            s_filter,
            self.file_type
        )
        return luigi.LocalTarget(fn)

class ObjectsScalesJointDist(luigi.Task):
    var1 = luigi.Parameter()
    var2 = luigi.Parameter()
    file_type = luigi.Parameter(default='png')

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
                    variables="{},{}".format(self.var1, self.var2),
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

    def run(self):
        inputs = self.input()

        for n, (base_name, input) in enumerate(inputs.items()):
            ds = input.open()
            da_v1 = ds[self.var1]
            da_v2 = ds[self.var2]

            desc = base_name.replace('_', ' ').replace('.', ' ')
            sns.jointplot(x=da_v1, y=da_v2, label=desc, s=10)

        plt.legend()

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
        fn = "objects_scales_joint_dist.{}__{}.{}{}.{}".format(
            self.var1,
            self.var2,
            self.base_names.replace(',', '__'),
            s_filter,
            self.file_type
        )
        return luigi.LocalTarget(fn)
