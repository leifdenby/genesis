import luigi

from .....objects.projected_2d.extraction import (
    get_approximate_cloud_underside,
    get_cloud_underside_for_new_formed_clouds,
)
from .....objects.projected_2d.utils import extract_from_3d_at_heights_in_2d
from .....utils import find_vertical_grid_spacing
from ..base import XArrayTarget, get_workdir
from ..extraction import ExtractField3D, TimeCrossSectionSlices2D
from . import uclales_2d_tracking
from .base import (
    PerformObjectTracking2D,
    TrackingType,
)


class EstimateCloudUndersideHeight(luigi.Task):
    """
    Extract from 3D field below clouds
    """

    base_name = luigi.Parameter()
    cloud_age_max = luigi.FloatParameter()
    ensure_tracked = luigi.BoolParameter(default=False)

    def requires(self):
        has_tracking = uclales_2d_tracking.HAS_TRACKING

        if not has_tracking and self.ensure_tracked:
            raise Exception(
                "To extract the below cloud-base state from only tracked clouds"
                " the 2D tracking utility must be available. Please set the"
                f"{uclales_2d_tracking.BIN_PATH_ENV_VAR} envirionment variable"
                " to point to the tracking utility"
            )

        reqs = {}
        reqs["qc"] = ExtractField3D(base_name=self.base_name, field_name="qc")
        if not self._use_approx_tracking():
            base_name_no_tn = self.base_name.split(".tn")[0]
            reqs["tracking"] = PerformObjectTracking2D(
                base_name=base_name_no_tn,
                tracking_type=TrackingType.CLOUD_CORE,
            )
            reqs["cldbase"] = TimeCrossSectionSlices2D(
                base_name=base_name_no_tn, var_name="cldbase"
            )
        return reqs

    def _use_approx_tracking(self):
        if self.ensure_tracked:
            return False

        return uclales_2d_tracking.HAS_TRACKING

    def run(self):
        if not self._use_approx_tracking():
            da_scalar_3d = self.input()["qc"].open()
            da_cloudbase_2d = self.input()["cldbase"].open()
            ds_tracking = self.input()["tracking"].open()

            t0 = da_scalar_3d.time.values[0]

            da_z_cu = get_cloud_underside_for_new_formed_clouds(
                ds_tracking=ds_tracking,
                da_cldbase_2d=da_cloudbase_2d,
                t0=t0,
                t_age_max=self.cloud_age_max,
            )
            method = "tracked clouds"
            num_clouds = da_z_cu.num_clouds

        else:
            qc = self.input()["qc"].open()
            da_z_cu = get_approximate_cloud_underside(qc=qc, z_tol=50.0)
            num_clouds = "unknown"
            method = "approximate"

        da_z_cu.attrs["method"] = method

        if not self._use_approx_tracking():
            da_z_cu.attrs["cloud_age_max"] = self.cloud_age_max
        da_z_cu.attrs["num_clouds"] = num_clouds
        da_z_cu.to_netcdf(self.output().fn)

    def output(self):
        name_parts = [
            self.base_name,
            "cloud_underside",
            self._use_approx_tracking() and "_approx" or "_tracked",
            "xy",
            "nc",
        ]

        if not self._use_approx_tracking():
            name_parts.insert(3, f"max_t_age__{self.cloud_age_max:.0f}s")

        fn = ".".join(name_parts)
        p = get_workdir() / self.base_name / fn
        return XArrayTarget(str(p))


class ExtractBelowCloudEnvironment(luigi.Task):
    """
    Extract from 3D field below clouds
    """

    base_name = luigi.Parameter()
    field_name = luigi.Parameter()
    cloud_age_max = luigi.FloatParameter()
    ensure_tracked = luigi.BoolParameter(default=False)

    def requires(self):
        reqs = dict(
            field=ExtractField3D(base_name=self.base_name, field_name=self.field_name),
            cloud_underside=EstimateCloudUndersideHeight(
                base_name=self.base_name,
                cloud_age_max=self.cloud_age_max,
                ensure_tracked=self.ensure_tracked,
            ),
        )
        return reqs

    def run(self):
        da_scalar_3d = self.input()["field"].open()
        da_z_cu = self.input()["cloud_underside"].open()

        dz = find_vertical_grid_spacing(da_scalar_3d)

        da_z_extract = da_z_cu - dz
        da_z_extract = da_z_extract.where(da_z_extract < 650)

        da_cu = extract_from_3d_at_heights_in_2d(
            da_3d=da_scalar_3d,
            z_2d=da_z_extract,
        )
        da_cu = da_cu.squeeze()
        da_cu.name = self.field_name
        da_cu.attrs.update(da_z_cu.attrs)
        da_cu.to_netcdf(self.output().fn)

    def output(self):
        use_approx_tracking = self.requires()["cloud_underside"]._use_approx_tracking()
        name_parts = [
            self.base_name,
            self.field_name,
            "cloud_underside",
            use_approx_tracking and "_approx" or "_tracked",
            "xy",
            "nc",
        ]

        if not use_approx_tracking:
            name_parts.insert(3, f"max_t_age__{self.cloud_age_max:.0f}s")

        fn = ".".join(name_parts)
        p = get_workdir() / self.base_name / fn
        return XArrayTarget(str(p))
