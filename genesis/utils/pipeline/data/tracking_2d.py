import warnings
import shutil
import os
from pathlib import Path

import luigi
import xarray as xr
import numpy as np
import dask_image.ndmeasure as dmeasure

from .... import objects
from .extraction import ExtractCrossSection2D, ExtractField3D
from .extraction import TimeCrossSectionSlices2D
from .base import get_workdir, _get_dataset_meta_info, XArrayTarget
from .masking import MakeMask
from ....bulk_statistics import cross_correlation_with_height
from ....utils import find_vertical_grid_spacing

from ..data_sources import uclales_2d_tracking
from ..data_sources.uclales_2d_tracking import TrackingType
from ..data_sources.uclales import _fix_time_units as fix_time_units
from ....objects.projected_2d import ObjectSet



class XArrayTargetUCLALES(XArrayTarget):
    def open(self, *args, **kwargs):
        kwargs['decode_times'] = False
        da = super().open(*args, **kwargs)
        da['time'], _ = fix_time_units(da['time'])
        if hasattr(da, 'to_dataset'):
            return xr.decode_cf(da.to_dataset())
        else:
            return xr.decode_cf(da)

class XArrayTargetUCLALESTracking(XArrayTarget):
    """
    The tracking file I have for the big RICO simulations have a lot of
    problems before we can load them CF-convention following data
    """

    @staticmethod
    def _fix_datetime(da):
        assert da.attrs['units'] == "day as %Y%m%d.%f"
        da.values = (da.values*24*60*60).astype(int)
        da.attrs['units'] = "seconds since 2000-01-01 00:00:00"
        return da

    @staticmethod
    def _fix_time_and_timedelta(da):
        assert da.attrs['units'] == "day as %Y%m%d.%f"
        da.values = (da.values*24*60*60).astype(int)
        da.attrs['units'] = "seconds"
        return da


    def open(self, *args, **kwargs):
        kwargs['decode_times'] = False
        ds = super().open(*args, **kwargs)

        datetime_vars = [
            "time",
            "smcloudtmax",
            "smcloudtmin",
            "smcoretmin",
            "smcoretmax",
        ]

        time_and_timedelta_vars = [
            "smcloudt",
            "smcoret",
            "smclouddur",
            "smcoredur",
        ]

        for v in datetime_vars:
            ds[v] = self._fix_datetime(ds[v])

        for v in time_and_timedelta_vars:
            ds[v] = self._fix_time_and_timedelta(ds[v])

        ds = xr.decode_cf(ds)

        # remove superflous indecies "smcloud" and "smcore"
        ds = ds.swap_dims(dict(smcloud="smcloudid", smcore="smcoreid"))

        return ds


class PerformObjectTracking2D(luigi.Task):
    base_name = luigi.Parameter()
    tracking_type = luigi.EnumParameter(enum=TrackingType)
    tn_start = luigi.Parameter(default=1)
    tn_end = luigi.Parameter(default=None)

    def requires(self):
        required_fields = uclales_2d_tracking.get_required_fields(
            tracking_type=self.tracking_type
        )

        return [
            TimeCrossSectionSlices2D(base_name=self.base_name,
                                     field_name=field_name)
            for field_name in required_fields
        ]

    def _get_tn_end(self):
        if self.tn_end is None:
            print(self.input()[0])
            da_input = self.input()[0].open()
            # TODO: use more intelligent selection for timesteps to track here
            tn_end = len(da_input.time)
        else:
            tn_end = self.tn_end
        return tn_end

    def _get_tn_start(self):
        tn_start = self.tn_start
        if tn_start != 1:
            warnings.warn("There is currently a bug in the cloud-tracking "
                          "code which causes it to crash when not starting "
                          "at time index 1. Setting tn_start=1")
            tn_start = 1
        return tn_start

    def _get_dataset_name(self):
        meta = _get_dataset_meta_info(self.base_name)
        return meta['experiment_name']

    def run(self):
        meta = _get_dataset_meta_info(self.base_name)
        # don't use base_name here because that will tie analysis to specific
        # timestamp, instead use `experiment_name`
        dataset_name = meta['experiment_name']

        p_source = Path(meta['path'])
        p_source_tracking = p_source/"tracking_output"

        # this shouln't be necessary, but... the fake file stuff below makes it
        # so
        if self.output().exists():
            return


        if 'tracking_2d' in meta:
            # tn_start, tn_end = meta['tracking_2d']['interval']
            # tracking_type = TrackingType["cloud,core,thermal"]
            p_source_tracking = p_source/"tracking_output"/meta['tracking_2d']

            if p_source_tracking.exists():
                Path(self.output().fn).parent.mkdir(parents=True, exist_ok=True)
                os.symlink(p_source_tracking, self.output().fn)

            return

        tn_start = self._get_tn_start()
        tn_end = self._get_tn_end()

        p_data = Path(self.input()[0].fn).parent
        fn_tracking = uclales_2d_tracking.call(
            data_path=p_data, dataset_name=dataset_name,
            tn_start=tn_start, tn_end=tn_end,
            tracking_type=self.tracking_type
        )

        shutil.move(fn_tracking, self.output().fn)

    def output(self):
        # if any([not input.exists() for input in self.input()]):
            # return luigi.LocalTarget('__fake__file__.nc')

        meta = _get_dataset_meta_info(self.base_name)
        type_id = uclales_2d_tracking.TrackingType.make_identifier(
            self.tracking_type
        )
        # interval_id = "tn{}_to_tn{}".format(
            # self._get_tn_start(), self._get_tn_end()
        # )

        interval_id = "__hardcoded_interval__"

        dataset_name = meta['experiment_name']
        FN_2D_FORMAT = ("{dataset_name}.tracking.{type_id}"
                        ".{interval_id}.out.xy.nc")

        fn = FN_2D_FORMAT.format(
            dataset_name=dataset_name, type_id=type_id,
            interval_id=interval_id
        )

        p = get_workdir()/self.base_name/"tracking_output"/fn
        return XArrayTargetUCLALESTracking(str(p))


class TrackingLabels2D(luigi.Task):
    base_name = luigi.Parameter()
    label_var = luigi.Parameter(default='nrcloud')
    reference_field = luigi.Parameter(default='qt')

    def requires(self):
        return dict(
            tracking=PerformObjectTracking2D(
                base_name=self.base_name,
                tracking_type=uclales_2d_tracking.TrackingType.CLOUD_CORE,
            ),
            ref_field=ExtractField3D(
                base_name=self.base_name,
                field_name=self.reference_field
            )
        )

    def run(self):
        da_timedep = self.input()['tracking'].open()
        da_3d_ref = self.input()['ref_field'].open()

        t0 = da_3d_ref.time

        da = da_timedep[self.label_var].sel(time=t0).squeeze()

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da.to_netcdf(self.output().fn)

    def output(self):
        fn = "{}.nc".format(self.label_var)
        p = get_workdir()/self.base_name/"tracking_labels_2d"/fn
        return XArrayTarget(str(p))


class Aggregate2DCrossSectionOnTrackedObjects(luigi.Task):
    field_name = luigi.Parameter()
    base_name = luigi.Parameter()
    reference_field = luigi.Parameter(default='qt')

    label_var = "nrcloud"

    def requires(self):
        return dict(
            tracking_labels=TrackingLabels2D(
                base_name=self.base_name,
                reference_field=self.reference_field,
            ),
            field=ExtractCrossSection2D(
                base_name=self.base_name,
                field_name=self.field_name,
                reference_field=self.reference_field,
            ),
        )

    def run(self):
        da_values = self.input()['field'].open()
        da_labels = self.input()['tracking_labels'].open().astype(int)

        if self.field_name in ["cldbase", "cldtop"]:
            dx = da_values.dz
        else:
            raise NotImplementedError(self.field_name)

        # set up bins for histogram
        v_min = da_values.min().item()
        v_max = da_values.max().item()
        hist_range = (v_min - dx*0.5, v_max + dx*0.5)
        nbins = int((hist_range[1] - hist_range[0])/dx)
        v_bins_c = np.linspace(v_min, v_max, nbins)

        # get unique object labels
        fn_unique_dropna = lambda v: np.unique(v.data[~np.isnan(v.data)])
        object_ids = fn_unique_dropna(da_labels)[1:]

        histogram = dmeasure.histogram(
            image=da_values,
            min=hist_range[0],
            max=hist_range[1],
            bins=nbins,
            label_image=da_labels,
            index=object_ids
        ).compute()

        values_binned = np.zeros((len(object_ids), nbins), dtype=int)
        for n in range(len(object_ids)):
            n_hist = len(histogram[n])
            values_binned[n][:n_hist] = histogram[n]

        bin_var = "{}__bin".format(self.field_name)

        da_binned = xr.DataArray(
            values_binned,
            dims=('object_id', bin_var),
            coords={
                'object_id': object_ids,
                bin_var: v_bins_c
            }
        )
        da_binned.coords[bin_var].attrs['units'] = da_values.units
        if 'longname' in da_values.attrs:
            # CF-conventions are `long_name` not `longname`
            da_binned.coords[bin_var].attrs['long_name'] = da_values.longname

        # bah xarray, "unsqueeze" copy over the squeeze again...
        da_binned = da_binned.expand_dims(dict(
            time=da_values.expand_dims('time').time
        )).squeeze()

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_binned.to_netcdf(self.output().fn)

    def output(self):
        fn = "{}.by_{}.nc".format(self.field_name, self.label_var)
        p = get_workdir()/self.base_name/"cross_sections"/"aggregated"/fn
        return XArrayTarget(str(p))


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

        t0 = mask.time
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
        if uclales_2d_tracking.HAS_TRACKING:
            return dict(
                field=ExtractField3D(
                    base_name=self.base_name,
                    field_name=self.field_name
                ),
                cldbase=ExtractCrossSection2D(
                    base_name=self.base_name,
                    field_name='cldbase',
                )
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
        if uclales_2d_tracking.HAS_TRACKING:
            ds_tracking = self.input()["tracking"].open()

            da_scalar_3d = self.input()["field"].open()

            dx = find_vertical_grid_spacing(da_scalar_3d)

            t0 = da_scalar_3d.time
            z_cb = cross_correlation_with_height.get_cloudbase_height(
                ds_tracking=ds_tracking, t0=t0, t_age_max=self.cloud_age_max,
                da_cldbase_2d=self.input()["cldbase"].open(), dx=dx
            )
            dz = find_vertical_grid_spacing(da_scalar_3d)
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


class ExtractNearCloudEnvironment(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    def requires(self):
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
        if uclales_2d_tracking.HAS_TRACKING:
            ds_tracking = self.input()["tracking"].open()

            da_scalar_3d = self.input()["field"].open()

            t0 = da_scalar_3d.time.values[0]
            z_cb = cross_correlation_with_height.get_cloudbase_height(
                ds_tracking=ds_tracking, t0=t0, t_age_max=self.cloud_age_max,
            )
            dz = find_vertical_grid_spacing(da_scalar_3d)
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
