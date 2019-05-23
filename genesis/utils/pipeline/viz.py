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
    mask = luigi.Parameter(default='no_mask')
    width_method = length_scales.cumulant.calc.WidthEstimationMethod.MASS_WEIGHTED

    def requires(self):
        def _fixname(v):
            if v == 'w':
                return 'w_zt'
            else:
                return v

        return [
                data.ExtractField3D(base_name=self.base_name,
                                    field_name=_fixname(self.v1)),
                data.ExtractField3D(base_name=self.base_name,
                                    field_name=_fixname(self.v2)),
        ]

    def run(self):
        da_v1 = self.input()[0].open(decode_times=False)
        da_v2 = self.input()[1].open(decode_times=False)

        calc_fn = length_scales.cumulant.vertical_profile.calc.get_height_variation_of_characteristic_scales

        da = calc_fn(
            v1_3d=da_v1, v2_3d=da_v2, width_method=self.width_method,
            z_max=self.z_max
        )

        da.to_netcdf(self.output().path)

    def output(self):
        fn = length_scales.cumulant.vertical_profile.calc.FN_FORMAT.format(
            base_name=self.base_name, v1=self.v1, v2=self.v2,
            mask=self.mask
        )
        return data.XArrayTarget(fn)

class CumulantScalesProfile(luigi.Task):
    base_names = luigi.Parameter()
    cumulants = luigi.Parameter()
    z_max = luigi.FloatParameter(default=700.)
    plot_type = luigi.Parameter(default='scales')
    filetype = luigi.Parameter(default='pdf')

    mask = 'no_mask'

    def _parse_cumulant_arg(self):
        cums = [c.split(':') for c in self.cumulants.split(',')]
        return [c for (n,c) in enumerate(cums) if cums.index(c) == n]

    def requires(self):
        reqs = {}

        for base_name in self.base_names.split(','):
            reqs[base_name] = [
                ExtractCumulantScaleProfile(
                    base_name=base_name, v1=c[0], v2=c[1]
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

        plt.savefig(self.output().path, bbox_inches='tight')

    def output(self):
        base_name = "__".join(self.base_names.split(','))
        fn = length_scales.cumulant.vertical_profile.plot.FN_FORMAT.format(
            base_name=base_name, plot_type=self.plot_type, mask=self.mask,
            filetype=self.filetype
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
        fig, axes = plt.subplots(nrows=len(base_names), ncols=2,
                                 sharex=True, sharey=True,
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
        ax.text(0.5, -0.3, ' ', transform=ax.transAxes)

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

        def _fixname(v):
            if v == 'w':
                return 'w_zt'
            else:
                return v

        return dict(
            (base_name, {
                self.v1 : data.ExtractField3D(field_name=_fixname(self.v1),
                                              base_name=base_name),
                self.v2 : data.ExtractField3D(field_name=_fixname(self.v2),
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

        plot_fn = length_scales.cumulant.sections.plot
        plot_fn(datasets=datasets, var_names=[self.v1, self.v2],)

        plt.savefig(self.output().fn, bbox_inches='tight')

    def output(self):
        fn = length_scales.cumulant.sections.FN_FORMAT_PLOT.format(
            v1=self.v1, v2=self.v2, filetype=self.filetype
        )

        return luigi.LocalTarget(fn)


class HorizontalMeanProfile(luigi.Task):
    base_name = luigi.Parameter()
    field_names = luigi.Parameter(default='qv,qc,t')

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

            print(v_mean.longname)

        sns.despine()
        plt.suptitle(title)

        plt.savefig(self.output().fn)

    def output(self):
        fn = "{base_name}.{variables}.mean_profile.pdf".format(
            base_name=self.base_name,
            variables="__".join(self._field_names)
        )
        return luigi.LocalTarget(fn)


class ObjectScales(luigi.Task):
    object_splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    variables = luigi.Parameter(default='com_angles')

    def requires(self):
        variables = self.variables.split(',')
        reqs = []

        MINKOWSKI_VARS = "length width thickness".split(" ")

        for v in variables:
            if v in MINKOWSKI_VARS:
                reqs.append(
                    data.ComputeObjectMinkowskiScales(
                        base_name=self.base_name,
                        object_splitting_scalar=self.object_splitting_scalar
                    )
                )
            else:
                reqs.append(
                    data.ComputeObjectScale(
                        base_name=self.base_name,
                        variable=v,
                        object_splitting_scalar=self.object_splitting_scalar
                    )
                )

        return reqs

    def run(self):
        ds = xr.merge([
            input.open(decode_times=False) for input in self.input()
        ])
        objects.topology.plots.minkowski_scales.main(ds=ds)
        plt.savefig(self.output().path)

    def output(self):
        fn = '{}.minkowski_scales.pdf'.format(self.base_name)
        return luigi.LocalTarget(fn)
