import matplotlib
matplotlib.use("Agg")

import luigi
import xarray as xr
import matplotlib.pyplot as plt

# from ...length_scales.plot import cumulant_scale_vs_height
from ...length_scales.process import cumulant_scale_vs_height
from ...bulk_statistics import cross_correlation_with_height
from ...length_scales.plot import cumulant_sections_vs_height as cumulant_sections_vs_height_plot

from . import data


class CumulantScalesProfile(luigi.Task):
    def requires(self):
        pass

    def run(self):
        cumulant_scale_vs_height.process
        cumulant_scale_vs_height.plot_default()
        pass

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

        main = cumulant_sections_vs_height_plot.main
        main(datasets=datasets, var_names=[self.v1, self.v2],)

        plt.savefig(self.output().fn, bbox_inches='tight')

    def output(self):
        fn = cumulant_sections_vs_height_plot.FN_FORMAT.format(
            v1=self.v1, v2=self.v2
        )

        return luigi.LocalTarget(fn)


