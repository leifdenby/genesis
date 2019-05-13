import itertools

import matplotlib
matplotlib.use("Agg")

import luigi
import xarray as xr
import matplotlib.pyplot as plt

# from ...length_scales.plot import cumulant_scale_vs_height
from ...bulk_statistics import cross_correlation_with_height

from ... import length_scales

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
            base_name=base_name, plot_type=self.plot_type, mask=self.mask
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

    def requires(self):
        reqs = dict(
            full_domain=[
                data.ExtractField3D(field_name=self.v1, base_name=self.base_name),
                data.ExtractField3D(field_name=self.v2, base_name=self.base_name),
            ],
            cloudbase=[
                data.ExtractCloudbaseState(base_name=self.base_name, field_name=self.v1),
                data.ExtractCloudbaseState(base_name=self.base_name, field_name=self.v2),
            ]
        )

        if self.mask is not None:
            reqs['mask'] = data.MakeMask(method_name=self.mask,
                                         method_extra_args=self.mask_args)

    def output(self):
        if self.mask is not None:
            raise NotImplementedError
            if not self.input()["mask"].exists():
                pass
            out_fn = '{}.cross_correlation.{}.{}.masked_by.{}.png'.format(
                self.base_name, self.v1, self.v2, mask_name
            )
        else:
            out_fn = '{}.cross_correlation.{}.{}.png'.format(
                self.base_name, self.v1, self.v2
            )
        return luigi.LocalTarget(out_fn)

    def run(self):
        ds_3d = xr.merge([
            xr.open_dataarray(r.fn) for r in self.input()["full_domain"]
        ])
        ds_cb = xr.merge([
            xr.open_dataarray(r.fn) for r in self.input()["cloudbase"]
        ])

        z_levels = (
            ds_3d.isel(zt=slice(None, None, self.dk))
                 .sel(zt=slice(0, self.z_max))
                 .zt)

        cross_correlation_with_height.main(ds_3d=ds_3d, z_levels=z_levels, ds_cb=ds_cb)

        plt.savefig(self.output().fn)

class CumulantSlices(luigi.Task):
    v1 = luigi.Parameter()
    v2 = luigi.Parameter()
    base_names = luigi.Parameter()

    z_step = luigi.IntParameter(default=4)
    z_max = luigi.FloatParameter(default=700.)

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
            v1=self.v1, v2=self.v2
        )

        return luigi.LocalTarget(fn)
