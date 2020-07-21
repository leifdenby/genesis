"""
On filtering:

There are two things we might apply filters to:
  1) the 3D object mask, or
  2) the object IDs
Use cases:

- extracting only thermals which have triggered clouds:
  - for this we want to remove all 3D objects which don't spatially overlap
    with tracked thermals which eventually triggered clouds (i.e. reached a
    certain height). And so we need:
      1) where the objects are defined in 3D by IdentifyObjects so that this
      may be projected to 2D
      2)

Approaches:

a) create a 2D mask of objects which were tracked from cloud tracking code and
   remove from the 3D mask of objects any points outside the 2D mask

b) try to assign each 3D object to a matching 2D tracked object. The two
   methods of object identification will come up with different object IDs so
   this is necessary. Could use height to eliminate some objects when doing the
   mapping since 2D cloud tracking knows height of object (approximately)

"""

import luigi

from .data import MakeMask, XArrayTarget, PerformObjectTracking2D
from ... import objects


class FilterTriggeringThermalsByMask(luigi.Task):
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")

    def requires(self):
        reqs = {}
        reqs["mask"] = MakeMask(
            base_name=self.base_name,
            method_name=self.mask_method,
            method_extra_args=self.mask_method_extra_args,
        )
        reqs["tracking"] = PerformObjectTracking2D(
            base_name=self.base_name,
            tracking_type=objects.filter.TrackingType.THERMALS_ONLY,
        )

        return reqs

    def run(self):
        input = self.input()
        mask = input["mask"].open(decode_times=False)
        cloud_data = self.requires()["tracking"].get_cloud_data()

        t0 = mask.time.values

        ds_track_2d = cloud_data._fh_track.sel(time=t0)
        objects_tracked_2d = ds_track_2d.nrthrm

        mask_filtered = mask.where(~objects_tracked_2d.isnull())

        mask_filtered.to_netcdf(mask_filtered)

    def output(self):
        fn = "triggering_thermals.nc"
        return XArrayTarget(fn)
