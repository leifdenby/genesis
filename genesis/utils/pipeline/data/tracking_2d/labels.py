from pathlib import Path

import luigi

from ...data_sources.uclales import tracking_2d as uclales_2d_tracking
from ..base import (
    NumpyDatetimeParameter,
    XArrayTarget,
    _get_dataset_meta_info,
    get_workdir,
)
from . import TrackingType
from .base import TrackingLabels2D, TrackingVariable2D


class DerivedLabels2D(luigi.Task):
    """
    Produce 2D label array at a specific time from tracked objects with
    specific properties (these conditions are implemented for each `label_type`)
    """

    label_type = luigi.Parameter(default="newlyformed_singlecore_clouds")

    base_name = luigi.Parameter()
    time = NumpyDatetimeParameter()

    tracking_type = luigi.EnumParameter(enum=TrackingType)
    tracking_timestep_interval = luigi.ListParameter(default=[])
    offset_labels_by_gal_transform = luigi.BoolParameter(default=False)
    track_without_gal_transform = luigi.BoolParameter(default=False)

    def requires(self):
        tasks = {}

        kws = dict(
            base_name=self.base_name,
            time=self.time,
            tracking_type=self.tracking_type,
            offset_labels_by_gal_transform=self.offset_labels_by_gal_transform,
            track_without_gal_transform=self.track_without_gal_transform,
            tracking_timestep_interval=self.tracking_timestep_interval,
        )

        if self.label_type == "newlyformed_singlecore_clouds":
            tasks["labels"] = TrackingLabels2D(
                label_var="cloud",
                *kws,
            )
            tasks["object_type"] = TrackingVariable2D(
                var_name="object_type",
                *kws,
            )
            tasks["object_age"] = TrackingVariable2D(
                var_name="object_age",
                *kws,
            )

        return tasks

    def run(self):
        if self.label_type == "newlyformed_singlecore_clouds":
            da_labels = self.input()["labels"].open().fillna(0).astype(int)
            raise NotImplementedError(da_labels)
            da_labels_filtered = None

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_labels_filtered.to_netcdf(self.output().fn)

    def output(self):
        type_id = uclales_2d_tracking.TrackingType.make_identifier(self.tracking_type)
        if self.tracking_timestep_interval:
            interval_id = "tn{}_to_tn{}".format(*self.tracking_timestep_interval)
        else:
            interval_id = "__all__"

        name_parts = [
            self.var_name,
            f"of_{self.label_var}",
            f"tracked_{type_id}",
            interval_id,
            self.time.isoformat(),
        ]

        if self.dx:
            name_parts.insert(1, f"{self.dx}_{self.op}")
        else:
            name_parts.insert(1, self.op)

        if self.offset_labels_by_gal_transform:
            meta = _get_dataset_meta_info(self.base_name)
            u_gal, v_gal = meta["U_gal"]
            name_parts.append(f"go_labels_{u_gal}_{v_gal}")

        if self.track_without_gal_transform:
            name_parts.append("go_track")

        fn = f"{'.'.join(name_parts)}.nc"

        p = get_workdir() / self.base_name / "cross_sections" / "aggregated" / fn
        return XArrayTarget(str(p))
