import luigi
import numpy as np

from ..base import XArrayTarget, get_workdir
from ..extraction import ExtractField3D


class DomainMeanVerticalFlux(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    def requires(self):
        return ExtractField3D(
            base_name=self.base_name,
            field_name=self.field_name,
        )

    def run(self):
        da_3d = input.input().open().squeeze()

        # calculate domain mean profile
        da_domain_mean_profile = da_3d.mean(
            dim=("xt", "yt"), dtype=np.float64, skipna=True
        )
        da_domain_mean_profile["sampling"] = "full domain"
        da_domain_mean_profile.to_netcdf(self.output().fn)

    def output(self):
        fn = "{base_name}.{field_name}__mean_flux.{filetype}".format(
            base_name=self.base_name,
            field_name=self.field_name,
            filetype="nc",
        )

        p = get_workdir() / self.base_name / fn
        target = XArrayTarget(str(p))
        return target
