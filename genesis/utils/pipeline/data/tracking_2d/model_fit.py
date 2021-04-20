import luigi
from tqdm import tqdm
import xarray as xr
from pathlib import Path

from . import (
    TrackingType,
    AllObjectsAll2DCrossSectionAggregations,
    TrackingVariable2D,
    uclales_2d_tracking,
)

from ..base import get_workdir, XArrayTarget

from .....objects.models import parcel_rise


class ParcelRiseModelFit(luigi.Task):
    base_name = luigi.Parameter()
    label_var = "cloud"
    var_name = "cldtop"
    use_relative_time_axis = luigi.BoolParameter(default=True)

    track_without_gal_transform = luigi.BoolParameter(default=False)
    tracking_type = luigi.EnumParameter(enum=TrackingType)
    tracking_timestep_interval = luigi.ListParameter([])

    def requires(self):
        common_kws = dict(
            base_name=self.base_name,
            tracking_type=TrackingType.CLOUD_CORE,
            track_without_gal_transform=self.track_without_gal_transform,
            tracking_timestep_interval=self.tracking_timestep_interval,
        )

        return dict(
            z=AllObjectsAll2DCrossSectionAggregations(
                label_var=self.label_var,
                var_name=self.var_name,
                op="maximum",
                **common_kws,
            ),
            cloudtype=TrackingVariable2D(
                var_name="smcloudtype",
                **common_kws,
            ),
        )

    def run(self):
        input = self.input()
        da = input["z"].open()
        da_cloudtype = input["cloudtype"].open()

        da_cloudtype = (
            da_cloudtype.rename(smcloudid="object_id").astype(int).drop("smcloud")
        )
        da_cloudtype["object_id"] = da_cloudtype.coords["object_id"].astype(int)

        single_cloud_ids = da_cloudtype.where(da_cloudtype == 2, drop=True).object_id

        datasets = []

        partial_fn = "__partial__run__.model_fit.nc"
        ds_partial = None
        if Path(partial_fn).exists():
            ds_partial = xr.open_dataset(partial_fn)
            single_cloud_ids = single_cloud_ids.sel(
                object_id=slice(max(ds_partial.object_id), None)
            )
            datasets.append(ds_partial)

        try:
            for cloud_id in tqdm(single_cloud_ids):
                da_obj_z = da.sel(object_id=cloud_id)
                ds_model_summary = parcel_rise.fit_model_and_summarise(
                    da_z=da_obj_z,
                    predictions="mean_with_quantiles",
                    var_name=self.var_name,
                )
                datasets.append(ds_model_summary)
        except Exception:
            ds = xr.concat(datasets, dim="object_id")
            if ds_partial is not None:
                ds_partial.close()
            ds.to_netcdf(partial_fn)
            raise

        ds = xr.concat(datasets, dim="object_id")
        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        ds.to_netcdf(self.output().fn)

    def output(self):
        type_id = uclales_2d_tracking.TrackingType.make_identifier(self.tracking_type)
        if self.tracking_timestep_interval:
            interval_id = "tn{}_to_tn{}".format(*self.tracking_timestep_interval)
        else:
            interval_id = "__all__"

        name_parts = [
            self.var_name,
            f"tracked_{type_id}",
            interval_id,
        ]

        if not self.use_relative_time_axis:
            name_parts.append("absolute_time")

        if self.track_without_gal_transform:
            name_parts.append("go_track")

        fn = f"{'.'.join(name_parts)}.nc"
        p = (
            get_workdir()
            / self.base_name
            / "tracking_2d"
            / "model_fit"
            / "parcel_rise"
            / fn
        )
        return XArrayTarget(str(p))
