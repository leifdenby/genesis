import matplotlib
matplotlib.use("Agg")

import luigi
import xarray as xr
import matplotlib.pyplot as plt

# from ...length_scales.plot import cumulant_scale_vs_height
from ...length_scales.process import cumulant_scale_vs_height
from ...bulk_statistics import cross_correlation_with_height

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

    def requires(self):
        return dict(
            full_domain=[
                data.ExtractField3D(field_name=self.v1, base_name=self.base_name),
                data.ExtractField3D(field_name=self.v2, base_name=self.base_name),
            ],
            cloudbase=[
                data.ExtractCloudbaseState(base_name=self.base_name, field_name=self.v1),
                data.ExtractCloudbaseState(base_name=self.base_name, field_name=self.v2),
            ]
        )

    def output(self):
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
