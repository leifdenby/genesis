import warnings
import shutil
import os
from pathlib import Path

import luigi
import xarray as xr
import numpy as np
import dask_image.ndmeasure as dmeasure
from tqdm import tqdm

from .... import objects
from .extraction import (
    ExtractCrossSection2D,
    ExtractField3D,
    TimeCrossSectionSlices2D,
    REGEX_INSTANTENOUS_BASENAME,
    remove_gal_transform,
)
from .base import get_workdir, _get_dataset_meta_info, XArrayTarget
from .base import NumpyDatetimeParameter
from .masking import MakeMask
from ....bulk_statistics import cross_correlation_with_height
from ....utils import find_vertical_grid_spacing
from ....objects.tracking_2d.family import create_tracking_family_2D_field

from ..data_sources import uclales_2d_tracking
from ..data_sources.uclales_2d_tracking import TrackingType
from ..data_sources.uclales import _fix_time_units as fix_time_units


class XArrayTargetUCLALES(XArrayTarget):
    def open(self, *args, **kwargs):
        kwargs["decode_times"] = False
        da = super().open(*args, **kwargs)
        da["time"], _ = fix_time_units(da["time"])
        if hasattr(da, "to_dataset"):
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
            should_be_datetime = to_fix["should_be_datetime"]
            vars = to_fix["vars"]

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
                    if np.max(da_old.values - da_old.astype(int).values) == 0 and np.max(da_old) > 1000.0:
                        warnings.warn(f"The units on `{da_old.name}` are given as"
                                      f" `{da_old.units}`, but all the values are"
                                      " integer value and the largest is > 1000"
                                      ", so the correct units will be assumed to"
                                      " be seconds.")
                        da_new = da_old.astype(int)
                    else:
                        # round to nearest second, some of the tracking files
                        # are stored as fraction of a day (...) and so when
                        # converting backing to seconds we sometimes get
                        # rounding errors
                        da_new = np.rint((da_old * 24 * 60 * 60)).astype(int)
                elif old_units == "seconds since 0-00-00 00:00:00":
                    da_new = da_old.copy()
                else:
                    raise NotImplementedError(old_units)

                da_new.attrs["units"] = new_units

                if v in ds.coords:
                    # as of xarray v0.15.1 we must use `assign_coords` instead
                    # of assigning directly to .values
                    ds = ds.assign_coords(**{v: da_new})
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
            kwargs["decode_times"] = False
            ds = super().open(*args, **kwargs)
            ds = self._fix_invalid_time_storage(ds=ds)

        extra_dims = ["smcloud", "smcore", "smthrm"]
        # remove superflous indecies "smcloud" and "smcore"
        for d in extra_dims:
            if d in ds:
                ds = ds.swap_dims({d: d + "id"})

        return ds


class PerformObjectTracking2D(luigi.Task):
    base_name = luigi.Parameter()
    tracking_type = luigi.EnumParameter(enum=TrackingType)
    timestep_interval = luigi.ListParameter(default=[])
    U_offset = luigi.ListParameter(default=[])

    def requires(self):
        if REGEX_INSTANTENOUS_BASENAME.match(self.base_name):
            raise Exception(
                "Shouldn't pass base_name with timestep suffix"
                " (`.tn`) to tracking util"
            )

        required_vars = uclales_2d_tracking.get_required_vars(
            tracking_type=self.tracking_type
        )

        return [
            TimeCrossSectionSlices2D(base_name=self.base_name, var_name=var_name)
            for var_name in required_vars
        ]

    def run(self):
        meta = _get_dataset_meta_info(self.base_name)

        if len(self.timestep_interval) == 0:
            tn_start = 0
            N_timesteps = {
                input.fn: int(input.open().time.count()) for input in self.input()
            }
            if len(set(N_timesteps.values())) == 1:
                tn_end = list(N_timesteps.values())[0] - 1
            else:
                s_files = "\n\t".join(
                    ["{fn}: {N}".format(fn=k, N=v) for (k, v) in N_timesteps.items()]
                )
                raise Exception(
                    "The input files required for tracking don't currently have"
                    " the same number of timesteps, maybe some of them need"
                    " recreating? Required files and number of timesteps:\n"
                    f"\n\t{s_files}"
                )
        else:
            tn_start, tn_end = self.timestep_interval

        if tn_start != 0:
            warnings.warn(
                "There is currently a bug in the cloud-tracking "
                "code which causes it to crash when not starting "
                "at time index 0 (fortran index 1). Setting "
                "tn_start=0"
            )
            tn_start = 0

        if meta.get("no_tracking_calls", False):
            filename = Path(self.output().fn).name
            p_source = Path(meta["path"])
            p_source_tracking = p_source / "tracking_output" / filename

            if p_source_tracking.exists():
                Path(self.output().fn).parent.mkdir(parents=True, exist_ok=True)
                os.symlink(p_source_tracking, self.output().fn)
            else:
                raise Exception("Automatic tracking calls have been disabled and"
                                f" couldn't find tracking output."
                                " Please run tracking utility externally and place output"
                                f" in `{p_source_tracking}`"
                                )

        else:
            dataset_name = meta["experiment_name"]

            p_data = Path(self.input()[0].fn).parent
            fn_tracking = uclales_2d_tracking.call(
                data_path=p_data,
                dataset_name=dataset_name,
                tn_start=tn_start + 1,
                tn_end=tn_end,
                tracking_type=self.tracking_type,
                U_offset=self.U_offset,
            )

            Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
            shutil.move(fn_tracking, self.output().fn)

    def output(self):
        type_id = uclales_2d_tracking.TrackingType.make_identifier(self.tracking_type)

        if len(self.timestep_interval) == 0:
            interval_id = "__all__"
        else:
            tn_start, tn_end = self.timestep_interval
            interval_id = "{}__{}".format(tn_start, tn_end)

        if self.U_offset:
            offset_s = "u{}_v{}_offset".format(*self.U_offset)
        else:
            offset_s = "no_offset"

        FN_2D_FORMAT = (
            "{base_name}.tracking.{type_id}" ".{interval_id}.{offset}.nc"
        )

        fn = FN_2D_FORMAT.format(
            base_name=self.base_name,
            type_id=type_id,
            interval_id=interval_id,
            offset=offset_s,
        )

        p = get_workdir() / self.base_name / "tracking_output" / fn
        return XArrayTargetUCLALESTracking(str(p))


class _Tracking2DExtraction(luigi.Task):
    """
    Base task for extracting fields from object tracking in 2D. This should
    never be called directly. Instead use either TrackingVariable2D or TrackingLabels2D
    """
    base_name = luigi.Parameter()
    track_without_gal_transform = luigi.BoolParameter(default=False)
    tracking_type = luigi.EnumParameter(enum=TrackingType)
    tracking_timestep_interval = luigi.ListParameter(default=[])

    def requires(self):
        U_tracking_offset = None
        if self.track_without_gal_transform:
            meta = _get_dataset_meta_info(self.base_name)
            U_tracking_offset = meta.get("U_gal", None)
            if U_tracking_offset is None:
                raise Exception(
                    "To remove the Galilean transformation before tracking"
                    " please define the transform velocity"
                    " as `U_gal` in datasources.yaml for"
                    " dataset `{}`".format(self.base_name)
                )
        return PerformObjectTracking2D(
            base_name=self.base_name, tracking_type=self.tracking_type,
            timestep_interval=self.tracking_timestep_interval,
            U_offset=U_tracking_offset
        )


class TrackingVariable2D(_Tracking2DExtraction):
    var_name = luigi.Parameter()

    def run(self):
        var_name = self.var_name
        da_input = self.input().open()

        if not var_name in da_input:
            available_vars = ", ".join(
                filter(lambda v: not v.startswith("nr"), list(da_input.data_vars))
            )
            raise Exception(
                f"Couldn't find the requested var `{self.var_name}`"
                f", available vars: {available_vars}"
            )
        da = da_input[var_name]

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da.to_netcdf(self.output().fn)

    def output(self):
        type_id = uclales_2d_tracking.TrackingType.make_identifier(self.tracking_type)
        if self.tracking_timestep_interval:
            interval_id = "{}_to_{}".format(*self.tracking_timestep_interval)
        else:
            interval_id = "__all__"

        name_parts = [
            self.var_name,
            f"tracked_{type_id}",
            interval_id,
        ]

        if self.track_without_gal_transform:
            name_parts.append("go_track")

        fn = f"{'.'.join(name_parts)}.nc"

        p = get_workdir() / self.base_name / "tracking_vars_2d" / fn
        return XArrayTarget(str(p))


class TrackingLabels2D(_Tracking2DExtraction):
    label_var = luigi.Parameter()
    time = NumpyDatetimeParameter()
    offset_labels_by_gal_transform = luigi.BoolParameter(default=False)


    def requires(self):
        if self.offset_labels_by_gal_transform and self.track_without_gal_transform:
            raise Exception("`offset_labels_by_gal_transform` and `track_without_gal_transform`"
                            " cannot both be true")

        if self.label_var == "cldthrm_family":
            return TrackingFamilyLabels2D(
                base_name=self.base_name,
            )
        else:
            return super(TrackingLabels2D, self).requires()

    def run(self):
        da_timedep = self.input().open()
        da_timedep["time"] = da_timedep.time

        if self.label_var == "cldthrm_family":
            label_var = self.label_var
        else:
            label_var = f"nr{self.label_var}"
            if not label_var in da_timedep:
                available_vars = ", ".join([
                    s.replace("nr", "(nr)") for s in
                    filter(lambda v: v.startswith("nr"), list(da_timedep.data_vars))
                ])
                raise Exception(
                    f"Couldn't find the requested label var `{self.label_var}`"
                    f", available vars: {available_vars}"
                )
            da_timedep = da_timedep[label_var]

        t0 = self.time
        da = da_timedep.sel(time=t0).squeeze()

        if self.offset_labels_by_gal_transform:
            tref = da_timedep.isel(time=0).time
            da = offset_labels_by_gal_transform(da=da, tref=tref, base_name=self.base_name)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da.to_netcdf(self.output().fn)

    def output(self):
        type_id = uclales_2d_tracking.TrackingType.make_identifier(self.tracking_type)
        if self.tracking_timestep_interval:
            interval_id = "{}_to_{}".format(*self.tracking_timestep_interval)
        else:
            interval_id = "__all__"

        name_parts = [
            f"nr{self.label_var}",
            f"tracked_{type_id}",
            interval_id,
            self.time.isoformat(),
        ]

        if self.offset_labels_by_gal_transform:
            name_parts.append("go_labels")

        if self.track_without_gal_transform:
            name_parts.append("go_track")

        fn = f"{'.'.join(name_parts)}.nc"

        p = get_workdir() / self.base_name / "tracking_labels_2d" / fn
        return XArrayTarget(str(p))


class Aggregate2DCrossSectionOnTrackedObjects(luigi.Task):
    """
    Produce per-tracked-object histogram of 2D variables, binning by for
    example the number of cells in each cloud `nrcloud`
    """

    var_name = luigi.Parameter()
    base_name = luigi.Parameter()
    time = NumpyDatetimeParameter()
    label_var = luigi.Parameter()
    op = luigi.Parameter()
    tracking_type = luigi.EnumParameter(enum=TrackingType)
    tracking_timestep_interval = luigi.ListParameter(default=[])
    offset_labels_by_gal_transform = luigi.BoolParameter(default=False)
    track_without_gal_transform = luigi.BoolParameter(default=False)

    dx = luigi.FloatParameter(default=-1.0)

    def requires(self):
        if self.op == "histogram" and (self.dx is None or self.dx <= 0.0):
            raise Exception("Op `histogram` requires `dx` parameter is set")

        tasks = dict(
            tracking_labels=TrackingLabels2D(
                base_name=self.base_name,
                time=self.time,
                label_var=self.label_var,
                tracking_type=self.tracking_type,
                offset_labels_by_gal_transform=self.offset_labels_by_gal_transform,
                track_without_gal_transform=self.track_without_gal_transform,
                tracking_timestep_interval=self.tracking_timestep_interval,
            ),
        )

        if self.var_name not in ["xt", "yt"]:
            tasks["field"] = ExtractCrossSection2D(
                base_name=self.base_name, var_name=self.var_name, time=self.time,
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
            hist_range = (v_min - dx * 0.5, v_max + dx * 0.5)
            nbins = int((hist_range[1] - hist_range[0]) / dx)
            v_bins_c = np.linspace(v_min, v_max, nbins)
            bin_var = da_values.name

            # get unique object labels
            def fn_unique_dropna(v):
                return np.unique(v.data[~np.isnan(v.data)])

            object_ids = fn_unique_dropna(da_labels)[1:]

            values_binned = np.zeros((len(object_ids), nbins), dtype=int)
            if len(object_ids) > 0:
                histogram = dmeasure.histogram(
                    image=da_values,
                    min=hist_range[0],
                    max=hist_range[1],
                    bins=nbins,
                    label_image=da_labels,
                    index=object_ids,
                ).compute()

                for n in range(len(object_ids)):
                    n_hist = len(histogram[n])
                    values_binned[n][:n_hist] = histogram[n]

            da_binned = xr.DataArray(
                values_binned,
                dims=("object_id", bin_var),
                coords={"object_id": object_ids, bin_var: v_bins_c},
            )

            da_binned.coords[bin_var].attrs["units"] = da_values.units
            if "long_name" in da_values.attrs:
                da_binned.coords[bin_var].attrs["long_name"] = da_values.long_name

        # bah xarray, "unsqueeze" copy over the squeeze again...
        da_binned = da_binned.expand_dims(dict(time=da_values.expand_dims("time").time))

        return da_binned

    def _aggregate_generic(self, da_values, da_labels, op):
        # get unique object labels
        def fn_unique_dropna(v):
            return np.unique(v.data[~np.isnan(v.data)])

        object_ids = fn_unique_dropna(da_labels)[1:]

        if len(object_ids) > 0:
            fn_op = getattr(dmeasure, op)

            values = fn_op(
                image=da_values, label_image=da_labels, index=object_ids
            ).compute()
        else:
            values = np.nan * np.ones(object_ids.shape)

        import ipdb

        with ipdb.launch_ipdb_on_exception():
            da = xr.DataArray(
                values, dims=("object_id"), coords={"object_id": object_ids},
            )

            if "long_name" in da_values:
                da.attrs["long_name"] = da_values.long_name + "__{}".format(op)

            if "units" in da_values:
                da.attrs["units"] = da_values.units

            # include time
            da = da.expand_dims(dict(time=da_values.expand_dims("time").time))

        return da

    def run(self):
        da_labels = self.input()["tracking_labels"].open().astype(int)

        if self.var_name in ["xt", "yt"]:
            if self.var_name in "xt":
                _, da_values = xr.broadcast(da_labels.xt, da_labels.yt)
            elif self.var_name in "yt":
                da_values, _ = xr.broadcast(da_labels.xt, da_labels.yt)
            else:
                raise NotImplementedError(self.var_name)
        else:
            da_values = self.input()["field"].open()

        if self.op == "histogram":
            da_out = self._aggregate_as_hist(da_values=da_values, da_labels=da_labels,)
        elif self.op == "mean":
            da_out = self._aggregate_generic(
                da_values=da_values, da_labels=da_labels, op=self.op
            )
        else:
            raise NotImplementedError(self.op)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_out.to_netcdf(self.output().fn)

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
            name_parts.append("go_labels")

        if self.track_without_gal_transform:
            name_parts.append("go_track")

        fn = f"{'.'.join(name_parts)}.nc"

        p = get_workdir() / self.base_name / "cross_sections" / "aggregated" / fn
        return XArrayTarget(str(p))


class AllObjectsAll2DCrossSectionAggregations(luigi.Task):
    """
    Combine aggregation output for all objects throughout their life-span
    """
    base_name = luigi.Parameter()
    label_var = luigi.Parameter()
    var_name = luigi.Parameter()
    op = luigi.Parameter()
    dx = luigi.FloatParameter(default=-1.0)

    track_without_gal_transform = luigi.BoolParameter(default=False)
    tracking_type = luigi.EnumParameter(enum=TrackingType)
    tracking_timestep_interval = luigi.ListParameter([])

    def requires(self):
        # first need to find duration of tracked object
        tasks = {}
        kwargs = dict(
            base_name=self.base_name,
            tracking_type=self.tracking_type,
            tracking_timestep_interval=self.tracking_timestep_interval,
            track_without_gal_transform=self.track_without_gal_transform,
        )
        
        tasks['t_start'] = TrackingVariable2D(
            var_name=f"sm{self.label_var}tmin",
            **kwargs
        )
        tasks['t_end'] = TrackingVariable2D(
            var_name=f"sm{self.label_var}tmax",
            **kwargs
        )
        tasks['t_global'] = TrackingVariable2D(
            var_name="time",
            **kwargs
        )
        return tasks

    def _build_agg_tasks(self):
        da_time = self.input()['t_global'].open()

        agg_tasks = {}
        for time in da_time.values:
            agg_task = Aggregate2DCrossSectionOnTrackedObjects(
                var_name=self.var_name,
                base_name=self.base_name,
                dx=self.dx,
                op=self.op,
                time=time,
                label_var=self.label_var,
                tracking_timestep_interval=self.tracking_timestep_interval,
                tracking_type=self.tracking_type,
                track_without_gal_transform=self.track_without_gal_transform,
            )
            agg_tasks[time] = agg_task
        return agg_tasks


    def run(self):
        agg_output = yield self._build_agg_tasks()

        inputs = self.input()
        da_time = inputs['t_global'].open()
        da_tstart = inputs['t_start'].open()
        da_tend = inputs['t_end'].open()
        dt = np.gradient(da_time.values)[0]

        obj_var = f"sm{self.label_var}id"
        all_object_ids = da_tstart[obj_var].astype(int)

        open_aggs = {}
        das_agg_objs = {}
        # iterate over the timesteps splitting the aggregation in each into
        # different lists for each object present
        for time in tqdm(da_time.values, desc="timestep", position=0):
            da_agg_all = agg_output[time].open()
            if "object_id" not in da_agg_all.coords:
                continue

            # find all objects that were present at this time
            import ipdb
            with ipdb.launch_ipdb_on_exception():
                object_ids = all_object_ids.where(
                    ((da_tstart <= time) * (time <= da_tend)), drop=True
                )

            for object_id in object_ids.values:
                try:
                    da_obj = da_agg_all.sel(object_id=object_id)
                except KeyError:
                    # NOTE: the tracking code has as bug so that tmax refers to the next
                    # timestep *after* unless the object exists in the very last
                    # timestep. We want to know which objects exist at the very last
                    # timestep examined, so we need to handle the indexing here and
                    # create a fake datasets with nans
                    da_obj = np.nan*da_agg_all.isel(object_id=0)
                    da_obj = da_obj.assign_coords(dict(object_id=object_id))

                if not object_id in das_agg_objs:
                    das_agg_objs[object_id] = []
                das_agg_objs[object_id].append(da_obj)

        # free up some memory
        del agg_output
        del open_aggs

        das_agg = []
        object_ids = sorted(list(das_agg_objs.keys()))
        for object_id in tqdm(object_ids, desc="object"):
            das_agg_obj = das_agg_objs[object_id]
            # have to fillna(0) because we're storing counts which are ints and
            # don't have a fill value
            da_agg_obj = xr.concat(das_agg_obj, dim="time").fillna(0)

            # check the start and end times match with what the cloud-tracking
            # code thinks...
            t_start_obj = da_tstart.sel({obj_var: object_id})
            assert t_start_obj == da_agg_obj.time.isel(time=0)
            t_end_obj = da_tend.sel({obj_var: object_id})
            if not t_end_obj == da_agg_obj.time.isel(time=-1):
                # NOTE: again the bug in the cloud-tracking code means that
                # objects that exist at the end of the time-range have their
                # end time actually *after* the very last timestep
                if t_end_obj == da_time.isel(time=-1) + dt:
                    pass
                else:
                    raise Exception("")

            da_time_relative = da_agg_obj.time - da_agg_obj.time.isel(time=0)
            da_time_relative_mins = (
                da_time_relative.dt.seconds/60.0 + 24.0*60.0*da_time_relative.dt.days
            )
            da_time_relative_mins.attrs['long_name'] = "time since forming"
            da_time_relative_mins.attrs['units'] = "min"
            da_agg_obj = (da_agg_obj
                .assign_coords(dict(time_relative=da_time_relative_mins))
                .swap_dims(dict(time="time_relative"))
            )

            das_agg.append(da_agg_obj)

        # fillna(0) required again
        da = xr.concat(das_agg, dim="object_id").fillna(0).astype(int)

        da.to_netcdf(self.output().fn)

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
        ]

        if self.track_without_gal_transform:
            name_parts.append("go_track")

        fn = f"{'.'.join(name_parts)}.nc"
        p = get_workdir() / self.base_name / "cross_sections" / "aggregated" / fn
        return XArrayTarget(str(p))

class Aggregate2DCrossSectionOnTrackedObjectsBy3DField(luigi.Task):
    base_name = luigi.Parameter()
    ref_field_name = luigi.Parameter(default="qt")
    label_var = luigi.Parameter()
    var_name = luigi.Parameter()

    def requires(self):
        assert REGEX_INSTANTENOUS_BASENAME.match(self.base_name)

        return ExtractField3D(base_name=self.base_name, field_name=self.ref_field_name,)

    def run(self):
        da_3d = self.input().open()

        t0 = da_3d.time
        dx = find_vertical_grid_spacing(da_3d)

        match = REGEX_INSTANTENOUS_BASENAME.match(self.base_name)
        base_name_2d = match.groupdict()["base_name_2d"]

        output = yield Aggregate2DCrossSectionOnTrackedObjects(
            var_name=self.var_name,
            base_name=base_name_2d,
            time=t0,
            label_var=self.label_var,
            dx=dx,
        )

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        os.symlink(output.fn, self.output().fn)

    def output(self):
        fn = "{}.by_{}.nc".format(self.var_name, self.label_var)
        p = get_workdir() / self.base_name / "cross_sections" / "aggregated" / fn
        return XArrayTarget(str(p))


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

        t0 = mask.time
        ds_track_2d = cloud_data._fh_track.sel(time=t0)
        objects_tracked_2d = ds_track_2d.nrthrm

        mask_filtered = mask.where(~objects_tracked_2d.isnull())

        mask_filtered.to_netcdf(mask_filtered)

    def output(self):
        fn = "triggering_thermals.mask.nc"
        p = get_workdir() / self.base_name / fn
        return XArrayTarget(str(p))


class ExtractCloudbaseState(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    cloud_age_max = luigi.FloatParameter(default=200.0)

    def requires(self):
        if uclales_2d_tracking.HAS_TRACKING:
            return dict(
                field=ExtractField3D(
                    base_name=self.base_name, field_name=self.field_name
                ),
                cldbase=ExtractCrossSection2D(
                    base_name=self.base_name, field_name="cldbase",
                ),
            )
        else:
            warnings.warn(
                "cloud tracking isn't available. Using approximate"
                " method for finding cloud-base height rather than"
                "tracking"
            )
            return dict(
                qc=ExtractField3D(base_name=self.base_name, field_name="qc"),
                field=ExtractField3D(
                    base_name=self.base_name, field_name=self.field_name
                ),
            )

    def run(self):
        if uclales_2d_tracking.HAS_TRACKING:
            ds_tracking = self.input()["tracking"].open()

            da_scalar_3d = self.input()["field"].open()

            dx = find_vertical_grid_spacing(da_scalar_3d)

            t0 = da_scalar_3d.time
            z_cb = cross_correlation_with_height.get_cloudbase_height(
                ds_tracking=ds_tracking,
                t0=t0,
                t_age_max=self.cloud_age_max,
                da_cldbase_2d=self.input()["cldbase"].open(),
                dx=dx,
            )
            dz = find_vertical_grid_spacing(da_scalar_3d)
            method = "tracked clouds"
        else:
            qc = self.input()["qc"].open()
            z_cb = cross_correlation_with_height.get_approximate_cloudbase_height(
                qc=qc, z_tol=50.0
            )
            da_scalar_3d = self.input()["field"].open()
            try:
                dz = find_vertical_grid_spacing(da_scalar_3d)
                method = "approximate"
            except Exception:
                warnings.warn(
                    "Using cloud-base state because vertical grid"
                    " spacing is non-uniform"
                )
                dz = 0.0
                method = "approximate, in-cloud"

        da_cb = cross_correlation_with_height.extract_from_3d_at_heights_in_2d(
            da_3d=da_scalar_3d, z_2d=z_cb - dz
        )
        da_cb = da_cb.squeeze()
        da_cb.name = self.field_name
        da_cb.attrs["method"] = method
        da_cb.attrs["cloud_age_max"] = self.cloud_age_max
        da_cb.attrs["num_clouds"] = z_cb.num_clouds

        da_cb.to_netcdf(self.output().fn)

    def output(self):
        fn = "{}.{}.max_t_age__{:.0f}s.cloudbase.xy.nc".format(
            self.base_name, self.field_name, self.cloud_age_max,
        )
        p = get_workdir() / self.base_name / fn
        return XArrayTarget(str(p))


class ExtractNearCloudEnvironment(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    def requires(self):
        return dict(
            qc=ExtractField3D(base_name=self.base_name, field_name="qc"),
            field=ExtractField3D(base_name=self.base_name, field_name=self.field_name),
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
            method = "tracked clouds"
        else:
            qc = self.input()["qc"].open()
            z_cb = cross_correlation_with_height.get_approximate_cloudbase_height(
                qc=qc, z_tol=50.0
            )
            da_scalar_3d = self.input()["field"].open()
            try:
                dz = find_vertical_grid_spacing(da_scalar_3d)
                method = "approximate"
            except Exception:
                warnings.warn(
                    "Using cloud-base state because vertical grid"
                    " spacing is non-uniform"
                )
                dz = 0.0
                method = "approximate, in-cloud"

        da_cb = cross_correlation_with_height.extract_from_3d_at_heights_in_2d(
            da_3d=da_scalar_3d, z_2d=z_cb - dz
        )
        da_cb = da_cb.squeeze()
        da_cb.name = self.field_name
        da_cb.attrs["method"] = method
        da_cb.attrs["cloud_age_max"] = self.cloud_age_max
        da_cb.attrs["num_clouds"] = z_cb.num_clouds

        da_cb.to_netcdf(self.output().fn)

    def output(self):
        fn = "{}.{}.max_t_age__{:.0f}s.cloudbase.xy.nc".format(
            self.base_name, self.field_name, self.cloud_age_max,
        )
        p = get_workdir() / self.base_name / fn
        return XArrayTarget(str(p))


class TrackingFamilyLabels2D(PerformObjectTracking2D):
    # we're going to be relating clouds and thermals, so need to have tracked
    # them both
    tracking_type = TrackingType.CLOUD_CORE_THERMAL
    def requires(self):
        return PerformObjectTracking2D(
            base_name=self.base_name,
            tracking_type=self.tracking_type,
            timestep_interval=self.timestep_interval,
            U_offset=self.U_offset,
        )

    def run(self):
        ds_tracking = self.input().open()
        da_family = create_tracking_family_2D_field(ds_tracking=ds_tracking)
        da_family.to_netcdf(self.output().fn)

    def output(self):
        p_tracking = Path(self.input().fn)
        base_name = self.base_name
        p_family = p_tracking.parent/p_tracking.name.replace(
            f"{base_name}.tracking.", f"{base_name}.tracking.cldthrm_family.",
        )
        return XArrayTarget(str(p_family))
