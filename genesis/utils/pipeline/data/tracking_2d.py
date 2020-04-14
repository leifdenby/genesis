from pathlib import Path
import re
import warnings

import luigi
import xarray as xr

from .... import objects
from .extraction import ExtractCrossSection2D, ExtractField3D
from .base import get_workdir, _get_dataset_meta_info, XArrayTarget
from .masking import MakeMask
from ....bulk_statistics import cross_correlation_with_height
from ....utils import find_vertical_grid_spacing


try:
    import cloud_tracking_analysis.cloud_data
    from cloud_tracking_analysis import CloudData, tracking_utility
    HAS_CLOUD_TRACKING = True
except ImportError:
    HAS_CLOUD_TRACKING = False


class PerformObjectTracking2D(luigi.Task):
    base_name = luigi.Parameter()
    tracking_type = luigi.EnumParameter(enum=objects.filter.TrackingType)

    def requires(self):
        if not HAS_CLOUD_TRACKING:
            raise Exception("cloud_tracking_analysis module isn't available")

        required_fields = tracking_utility.get_required_fields(
            tracking_type=self.tracking_type
        )

        return [
            ExtractCrossSection2D(base_name=self.base_name,
                                  field_name=field_name)
            for field_name in required_fields
        ]

    def _get_tracking_identifier(self, meta):
        # timestep_3d = meta['timestep']

        da_input = xr.open_dataarray(self.input()[0].fn, decode_times=False)
        # TODO: use more intelligent selection for timesteps to track here
        tn_max = len(da_input.time)

        return "track_1-{}".format(tn_max)

    def _get_dataset_name(self):
        meta = _get_dataset_meta_info(self.base_name)
        return meta['experiment_name']

    def get_cloud_data(self):
        meta = _get_dataset_meta_info(self.base_name)
        dataset_name = meta['experiment_name']

        tracking_identifier = self._get_tracking_identifier(meta)

        p_data = get_workdir()
        cloud_tracking_analysis.cloud_data.ROOT_DIR = str(p_data)
        cloud_data = CloudData(dataset_name, tracking_identifier,
                               dataset_pathname=self.base_name,
                               tracking_type=self.tracking_type)

        return cloud_data

    def run(self):
        self.get_cloud_data()

    def output(self):
        if not all([i.exists() for i in self.input()]):
            return luigi.LocalTarget("fakefile.nc")

        meta = _get_dataset_meta_info(self.base_name)
        tracking_identifier = self._get_tracking_identifier(meta)
        tracking_identifier = tracking_identifier.replace(
            '_',
            '__{}__'.format(
                tracking_utility.TrackingType.make_identifier(
                    self.tracking_type
                )
            )
        )

        dataset_name = meta['experiment_name']
        FN_2D_FORMAT = "{}.out.xy.{}.nc"

        fn = FN_2D_FORMAT.format(dataset_name, tracking_identifier)
        p = get_workdir()/self.base_name/"tracking_output"/fn
        return luigi.LocalTarget(str(p))


class FilterTriggeringThermalsByMask(luigi.Task):
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')

    def requires(self):
        reqs = {}
        reqs['mask'] = MakeMask(
            base_name=self.base_name,
            method_name=self.mask_method,
            method_extra_args=self.mask_method_extra_args
        )
        reqs['tracking'] = PerformObjectTracking2D(
            base_name=self.base_name,
            tracking_type=objects.filter.TrackingType.THERMALS_ONLY
        )

        return reqs

    def run(self):
        input = self.input()
        mask = input['mask'].open(decode_times=False)
        cloud_data = self.requires()['tracking'].get_cloud_data()

        t0 = mask.time.values

        ds_track_2d = cloud_data._fh_track.sel(time=t0)
        objects_tracked_2d = ds_track_2d.nrthrm

        mask_filtered = mask.where(~objects_tracked_2d.isnull())

        mask_filtered.to_netcdf(mask_filtered)

    def output(self):
        fn = 'triggering_thermals.mask.nc'
        p = get_workdir()/self.base_name/fn
        return XArrayTarget(str(p))


class ExtractCloudbaseState(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    cloud_age_max = luigi.FloatParameter(default=200.)

    def requires(self):
        if HAS_CLOUD_TRACKING:
            return dict(
                tracking=PerformObjectTracking2D(
                    base_name=self.base_name
                ),
                field=ExtractField3D(
                    base_name=self.base_name,
                    field_name=self.field_name
                ),
            )
        else:
            warnings.warn("cloud tracking isn't available. Using approximate"
                          " method for finding cloud-base height rather than"
                          "tracking")
            return dict(
                qc=ExtractField3D(
                    base_name=self.base_name,
                    field_name='qc'
                ),
                field=ExtractField3D(
                    base_name=self.base_name,
                    field_name=self.field_name
                ),
            )

    def run(self):
        if HAS_CLOUD_TRACKING:
            tracking_output = Path(self.input()["tracking"].fn)
            TRACK_FN_RE = "(.*).out.xy.track__cloud_core__(.*).nc"
            matches = re.match(TRACK_FN_RE, tracking_output.name)
            dataset_name = matches[1]
            tracking_timerange = matches[2]
            tracking_identifier = "track_{}".format(tracking_timerange)

            p_data = get_workdir()
            cloud_tracking_analysis.cloud_data.ROOT_DIR = str(p_data)
            cloud_data = CloudData(dataset_name, tracking_identifier,
                                   dataset_pathname=self.base_name)

            da_scalar_3d = xr.open_dataarray(
                self.input()["field"].fn, decode_times=False
            )

            t0 = da_scalar_3d.time.values[0]
            z_cb = cross_correlation_with_height.get_cloudbase_height(
                cloud_data=cloud_data, t0=t0, t_age_max=self.cloud_age_max,
            )
            dz = cloud_data.dx
            method = 'tracked clouds'
        else:
            qc = self.input()['qc'].open()
            z_cb = (
                cross_correlation_with_height
                .get_approximate_cloudbase_height(
                    qc=qc, z_tol=50.
                )
            )
            da_scalar_3d = self.input()['field'].open()
            try:
                dz = find_vertical_grid_spacing(da_scalar_3d)
                method = 'approximate'
            except Exception:
                warnings.warn("Using cloud-base state because vertical grid"
                              " spacing is non-uniform")
                dz = 0.0
                method = 'approximate, in-cloud'

        da_cb = cross_correlation_with_height.extract_from_3d_at_heights_in_2d(
            da_3d=da_scalar_3d, z_2d=z_cb-dz
        )
        da_cb = da_cb.squeeze()
        da_cb.name = self.field_name
        da_cb.attrs['method'] = method
        da_cb.attrs['cloud_age_max'] = self.cloud_age_max
        da_cb.attrs['num_clouds'] = z_cb.num_clouds

        da_cb.to_netcdf(self.output().fn)

    def output(self):
        fn = "{}.{}.max_t_age__{:.0f}s.cloudbase.xy.nc".format(
            self.base_name, self.field_name, self.cloud_age_max,
        )
        p = get_workdir()/self.base_name/fn
        return XArrayTarget(str(p))
