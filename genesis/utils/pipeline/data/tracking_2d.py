import warnings
import shutil
import os
from pathlib import Path
import re

import luigi
import xarray as xr
import numpy as np
import dask_image.ndmeasure as dmeasure

from .... import objects
from .extraction import (
    ExtractCrossSection2D, ExtractField3D, TimeCrossSectionSlices2D,
    REGEX_INSTANTENOUS_BASENAME, remove_gal_transform
)
from .base import get_workdir, _get_dataset_meta_info, XArrayTarget
from .base import NumpyDatetimeParameter
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
    def _fix_invalid_time_storage(ds):
        datetime_vars = [
            "time",
            "smcloudtmax",
            "smcloudtmin",
            "smcoretmin",
            "smcoretmax",
            "smthrmtmin",
            "smthrmtmax",
        ]

        time_and_timedelta_vars = [
            "smcloudt",
            "smcoret",
            "smthrmt",
            "smclouddur",
            "smcoredur",
            "smthrmdur",
        ]

        variables_to_fix = [
            dict(should_be_datetime=True, vars=datetime_vars),
            dict(should_be_datetime=False, vars=time_and_timedelta_vars),
        ]

        for to_fix in variables_to_fix:
            should_be_datetime = to_fix['should_be_datetime']
            vars = to_fix['vars']

            for v in vars:
                if v not in ds:
                    continue

                da_old = ds[v]
                old_units = da_old.units

                if should_be_datetime:
                    new_units = "seconds since 2000-01-01 00:00:00"
                else:
                    new_units = "seconds"

                if old_units == "day as %Y%m%d.%f":
                    # round to nearest second, some of the tracking files
                    # are stored as fraction of a day (...) and so when
                    # converting backing to seconds we sometimes get
                    # rounding errors
                    da_new = np.rint((da_old*24*60*60)).astype(int)
                elif old_units == "seconds since 0-00-00 00:00:00":
                    da_new = da_old.copy()
                else:
                    raise NotImplementedError(old_units)

                da_new.attrs['units'] = new_units

                if v in ds.coords:
                    # as of xarray v0.15.1 we must use `assign_coords` instead
                    # of assigning directly to .values
                    ds = ds.assign_coords(**{ v: da_new })
                else:
                    ds[v] = da_new

        import ipdb
        with ipdb.launch_ipdb_on_exception():
            return xr.decode_cf(ds)

    def open(self, *args, **kwargs):
        try:
            ds = super().open(*args, **kwargs)
            if not np.issubdtype(ds.time.dtype, np.datetime64):
                convert_times = True
        except ValueError:
            convert_times = True

        if convert_times:
            kwargs['decode_times'] = False
            ds = super().open(*args, **kwargs)
            ds = self._fix_invalid_time_storage(ds=ds)

        extra_dims = ["smcloud", "smcore", "smthrm"]
        # remove superflous indecies "smcloud" and "smcore"
        for d in extra_dims:
            if d in ds:
                ds = ds.swap_dims({ d: d+"id" })

        return ds


class PerformObjectTracking2D(luigi.Task):
    base_name = luigi.Parameter()
    tracking_type = luigi.EnumParameter(enum=TrackingType)
    timestep_interval = luigi.ListParameter(default=[])

    def requires(self):
        if REGEX_INSTANTENOUS_BASENAME.match(self.base_name):
            raise Exception("Shouldn't pass base_name with timestep suffix"
                            " (`.tn`) to tracking util")

        required_fields = uclales_2d_tracking.get_required_fields(
            tracking_type=self.tracking_type
        )

        return [
            TimeCrossSectionSlices2D(base_name=self.base_name,
                                     field_name=field_name)
            for field_name in required_fields
        ]

    def run(self):
        meta = _get_dataset_meta_info(self.base_name)

        if len(self.timestep_interval) == 0:
            tn_start = 0
            da_input = self.input()[0].open()
            tn_end = len(da_input.time)-1
        else:
            tn_start, tn_end = self.timestep_interval

        if tn_start != 0:
            warnings.warn("There is currently a bug in the cloud-tracking "
                          "code which causes it to crash when not starting "
                          "at time index 0 (fortran index 1). Setting "
                          "tn_start=0")
            tn_start = 0


        if 'tracking_2d' in meta:
            # tn_start, tn_end = meta['tracking_2d']['interval']
            # tracking_type = TrackingType["cloud,core,thermal"]

            p_source = Path(meta['path'])
            p_source_tracking = p_source/"tracking_output"/meta['tracking_2d']

            if p_source_tracking.exists():
                Path(self.output().fn).parent.mkdir(parents=True, exist_ok=True)
                os.symlink(p_source_tracking, self.output().fn)

        else:
            dataset_name = meta['experiment_name']

            p_data = Path(self.input()[0].fn).parent
            fn_tracking = uclales_2d_tracking.call(
                data_path=p_data, dataset_name=dataset_name,
                tn_start=tn_start+1, tn_end=tn_end,
                tracking_type=self.tracking_type
            )

            Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
            shutil.move(fn_tracking, self.output().fn)

    def output(self):
        meta = _get_dataset_meta_info(self.base_name)
        type_id = uclales_2d_tracking.TrackingType.make_identifier(
            self.tracking_type
        )

        if len(self.timestep_interval) == 0:
            interval_id = "__all__"
        else:
            tn_start, tn_end = self.timestep_interval
            interval_id = "{}__{}".format(tn_start, tn_end)

        FN_2D_FORMAT = ("{base_name}.tracking.{type_id}"
                        ".{interval_id}.out.xy.nc")

        fn = FN_2D_FORMAT.format(
            base_name=self.base_name, type_id=type_id,
            interval_id=interval_id
        )

        p = get_workdir()/self.base_name/"tracking_output"/fn
        return XArrayTargetUCLALESTracking(
            str(p)
        )


class TrackingLabels2D(luigi.Task):
    base_name = luigi.Parameter()
    label_var = luigi.Parameter()
    time = NumpyDatetimeParameter()
    remove_gal_transform = luigi.BoolParameter(default=False)
    tracking_type = luigi.EnumParameter(enum=TrackingType)

    def requires(self):
        return PerformObjectTracking2D(
            base_name=self.base_name,
            tracking_type=self.tracking_type,
        )

    def run(self):
        da_timedep = self.input().open()
        da_timedep['time'] = da_timedep.time

        t0 = self.time
        da = da_timedep[self.label_var].sel(time=t0).squeeze()

        if self.remove_gal_transform:
            tref = da_timedep.isel(time=0).time
            da = remove_gal_transform(da=da, tref=tref,
                                      base_name=self.base_name)


        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da.to_netcdf(self.output().fn)

    def output(self):
        fn = "{}.{}_gal_transform.{}.nc".format(
            self.label_var,
            ["with", "without"][self.remove_gal_transform],
            self.time.isoformat(),
        )
        p = get_workdir()/self.base_name/"tracking_labels_2d"/fn
        return XArrayTarget(str(p))


class Aggregate2DCrossSectionOnTrackedObjects(luigi.Task):
    """
    Produce per-tracked-object histogram of 2D variables, binning by for
    example the number of cells in each cloud `nrcloud`
    """
    field_name = luigi.Parameter()
    base_name = luigi.Parameter()
    time = NumpyDatetimeParameter()
    label_var = luigi.Parameter()
    op = luigi.Parameter()
    tracking_type = luigi.EnumParameter(enum=TrackingType)
    remove_gal_transform = luigi.BoolParameter(default=False)

    dx = luigi.FloatParameter(default=None)

    def requires(self):
        if self.op == 'histogram' and self.dx is None:
            raise Exception("Op `histogram` requires `dx` parameter is set")

        tasks = dict(
            tracking_labels=TrackingLabels2D(
                base_name=self.base_name,
                time=self.time,
                label_var=self.label_var,
                tracking_type=self.tracking_type,
                remove_gal_transform=self.remove_gal_transform
            ),
        )

        if not self.field_name in ['xt', 'yt']:
            tasks['field'] = ExtractCrossSection2D(
                base_name=self.base_name,
                field_name=self.field_name,
                time=self.time,
            )

        return tasks

    def _aggregate_as_hist(self, da_values, da_labels):
        dx = self.dx

        # set up bins for histogram
        v_min = da_values.min().item()
        v_max = da_values.max().item()

        # if the extremes are nan we don't have any non-nan values, so we can't
        # make a histogram
        if np.isnan(v_max) or np.isnan(v_min):
            object_ids = []
            v_bins_c = []
            values_binned = [[]]

            da_binned = xr.DataArray()
        else:
            hist_range = (v_min - dx*0.5, v_max + dx*0.5)
            nbins = int((hist_range[1] - hist_range[0])/dx)
            v_bins_c = np.linspace(v_min, v_max, nbins)

            # get unique object labels
            fn_unique_dropna = lambda v: np.unique(v.data[~np.isnan(v.data)])
            object_ids = fn_unique_dropna(da_labels)[1:]

            values_binned = np.zeros((len(object_ids), nbins), dtype=int)
            if len(object_ids) > 0:
                histogram = dmeasure.histogram(
                    image=da_values,
                    min=hist_range[0],
                    max=hist_range[1],
                    bins=nbins,
                    label_image=da_labels,
                    index=object_ids
                ).compute()

                for n in range(len(object_ids)):
                    n_hist = len(histogram[n])
                    values_binned[n][:n_hist] = histogram[n]

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
        ))

        return da_binned


    def _aggregate_generic(self, da_values, da_labels, op):
        # get unique object labels
        fn_unique_dropna = lambda v: np.unique(v.data[~np.isnan(v.data)])
        object_ids = fn_unique_dropna(da_labels)[1:]

        if len(object_ids) > 0:
            fn_op = getattr(dmeasure, op)

            values = fn_op(
                image=da_values,
                label_image=da_labels,
                index=object_ids
            ).compute()
        else:
            values = np.nan*np.ones(object_ids.shape)

        import ipdb
        with ipdb.launch_ipdb_on_exception():
            da = xr.DataArray(
                values,
                dims=('object_id'),
                coords={
                    'object_id': object_ids,
                },
            )

            if 'long_name' in da_values:
                da.attrs['long_name'] = da_values.long_name + '__{}'.format(op)

            if 'units' in da_values:
                da.attrs['units'] = da_values.units

            # include time
            da = da.expand_dims(dict(
                time=da_values.expand_dims('time').time
            ))

        return da

    def run(self):
        da_labels = self.input()['tracking_labels'].open().astype(int)

        if self.field_name in ['xt', 'yt']:
            if self.field_name in 'xt':
                _, da_values = xr.broadcast(da_labels.xt, da_labels.yt)
            elif self.field_name in 'yt':
                da_values, _ = xr.broadcast(da_labels.xt, da_labels.yt)
            else:
                raise NotImplementedError(self.field_name)
        else:
            da_values = self.input()['field'].open()

        if self.op == 'histogram':
            da_out = self._aggregate_as_hist(
                da_values=da_values, da_labels=da_labels,
            )
        elif self.op == 'mean':
            da_out = self._aggregate_generic(
                da_values=da_values, da_labels=da_labels, op=self.op
            )
        else:
            raise NotImplementedError(self.op)


        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_out.to_netcdf(self.output().fn)

    def output(self):
        fn = "{}__{}.{}_gal_transform.by_{}.at_{}.nc".format(
            self.field_name, self.op,
            ["with", "without"][self.remove_gal_transform],
            self.label_var, self.time.isoformat()
        )
        p = get_workdir()/self.base_name/"cross_sections"/"aggregated"/fn
        return XArrayTarget(str(p))


class AggregateAll2DCrossSectionOnTrackedObjects(luigi.Task):
    field_name = luigi.Parameter()
    base_name = luigi.Parameter()
    dx = luigi.FloatParameter()
    label_var = luigi.Parameter()
    tn_min = luigi.IntParameter(default=20)
    tn_max = luigi.IntParameter(default=50)

    def requires(self):
        return TimeCrossSectionSlices2D(
            base_name=self.base_name,
            field_name=self.field_name
        )

    def _build_tasks(self):
        da_timedep = self.input().open()
        tasks = []
        for t in da_timedep.time.values:
            t = Aggregate2DCrossSectionOnTrackedObjects(
                field_name=self.field_name,
                base_name=self.base_name,
                dx=self.dx,
                label_var=self.label_var,
                time=t,
            )
            tasks.append(t)
        return tasks[self.tn_min:self.tn_max]

    def run(self):
        inputs = yield self._build_tasks()
        ds = xr.open_mfdataset(
            [input.fn for input in inputs], combine='nested',
            concat_dim='time'
        )

        ds.to_netcdf(self.output().fn)

    def output(self):
        fn = "{}.by_{}.tn{}_to_tn{}.nc".format(
            self.field_name, self.label_var, self.tn_min, self.tn_max
        )
        p = get_workdir()/self.base_name/"cross_sections"/"aggregated"/fn
        return XArrayTarget(str(p))


class Aggregate2DCrossSectionOnTrackedObjectsBy3DField(luigi.Task):
    base_name = luigi.Parameter()
    ref_field_name = luigi.Parameter(default='qt')
    label_var = luigi.Parameter()
    field_name = luigi.Parameter()

    def requires(self):
        assert REGEX_INSTANTENOUS_BASENAME.match(self.base_name)

        return ExtractField3D(
            base_name=self.base_name,
            field_name=self.ref_field_name,
        )

    def run(self):
        da_3d = self.input().open()

        t0 = da_3d.time
        dx = find_vertical_grid_spacing(da_3d)

        match = REGEX_INSTANTENOUS_BASENAME.match(self.base_name)
        base_name_2d = match.groupdict()['base_name_2d']

        output = yield Aggregate2DCrossSectionOnTrackedObjects(
            field_name=self.field_name,
            base_name=base_name_2d,
            time=t0,
            label_var=self.label_var,
            dx=dx
        )

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        os.symlink(output.fn, self.output().fn)

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
