import datetime
import os
from hashlib import md5
from pathlib import Path

import dask_image.ndmeasure as dmeasure
import luigi
import numpy as np
import xarray as xr
from tqdm import tqdm

from .....objects.projected_2d import CloudType
from .....objects.projected_2d.aggregation import \
    cloudbase_max_height_by_histogram_peak
from .....utils import find_horizontal_grid_spacing, find_vertical_grid_spacing
from ...data_sources.uclales import tracking_2d as uclales_2d_tracking
from ..base import (NumpyDatetimeParameter, XArrayTarget,
                    _get_dataset_meta_info, get_workdir)
from ..extraction import (REGEX_INSTANTENOUS_BASENAME, ExtractCrossSection2D,
                          ExtractField3D, TimeCrossSectionSlices2D,
                          remove_gal_transform)
from . import TrackingType
from .base import TrackingLabels2D, TrackingVariable2D

N_parallel_tasks = 100000


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

        if self.var_name not in ["xt", "yt", "area"]:
            tasks["field"] = ExtractCrossSection2D(
                base_name=self.base_name,
                var_name=self.var_name,
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

        da = xr.DataArray(
            values,
            dims=("object_id"),
            coords={"object_id": object_ids},
        )

        if "long_name" in da_values:
            da.attrs["long_name"] = da_values.long_name + "__{}".format(op)

        if "units" in da_values:
            da.attrs["units"] = da_values.units

        # include time
        da = da.expand_dims(dict(time=da_values.expand_dims("time").time))

        return da

    def run(self):
        da_labels = self.input()["tracking_labels"].open().fillna(0).astype(int)

        if self.var_name in ["xt", "yt"]:
            if self.var_name in "xt":
                _, da_values = xr.broadcast(da_labels.xt, da_labels.yt)
            elif self.var_name in "yt":
                da_values, _ = xr.broadcast(da_labels.xt, da_labels.yt)
            else:
                raise NotImplementedError(self.var_name)

            # (x, y) values change due to the Galliean transform, so we need to
            # use the actual translated grid positions
            if self.offset_labels_by_gal_transform:
                tref = da_labels.time
                da_values = remove_gal_transform(
                    da=da_values, tref=tref, base_name=self.base_name
                )
        elif self.var_name == "area":
            meta = _get_dataset_meta_info(self.base_name)
            dx = find_horizontal_grid_spacing(da_labels)
            da_values = xr.ones_like(da_labels)
            da_values.attrs["units"] = f"{da_labels.xt.units}^2"
            da_values.attrs["long_name"] = "area"
        else:
            da_values = self.input()["field"].open()

        if self.op == "histogram":
            da_out = self._aggregate_as_hist(
                da_values=da_values,
                da_labels=da_labels,
            )
            name = f"{self.var_name}__hist_{self.dx}"
        elif self.op in vars(dmeasure):
            da_out = self._aggregate_generic(
                da_values=da_values, da_labels=da_labels, op=self.op
            )
            name = f"{self.var_name}__{self.op}"
        else:
            raise NotImplementedError(self.op)

        da_out.name = name

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
            meta = _get_dataset_meta_info(self.base_name)
            u_gal, v_gal = meta["U_gal"]
            name_parts.append(f"go_labels_{u_gal}_{v_gal}")

        if self.track_without_gal_transform:
            name_parts.append("go_track")

        fn = f"{'.'.join(name_parts)}.nc"

        p = get_workdir() / self.base_name / "cross_sections" / "aggregated" / fn
        return XArrayTarget(str(p))


class Object2DCrossSectionAggregation(luigi.Task):
    """
    Combine aggregation output for a single object throughout its life-span
    """

    base_name = luigi.Parameter()
    label_var = luigi.OptionalParameter(default=None)
    var_name = luigi.Parameter()
    op = luigi.OptionalParameter(default=None)
    dx = luigi.FloatParameter(default=-1.0)
    use_relative_time_axis = luigi.BoolParameter(default=True)

    track_without_gal_transform = luigi.BoolParameter(default=False)
    tracking_type = luigi.EnumParameter(enum=TrackingType)
    tracking_timestep_interval = luigi.ListParameter([])
    times = luigi.ListParameter()
    object_id = luigi.IntParameter()

    @property
    def _label_var(self):
        if self.var_name == "z_cb_est":
            if self.label_var is [None, "None"]:
                raise Exception(
                    "Shouldn't provide `label_var` when estimating cloud-base height"
                )
            label_var = "cloud"
        else:
            label_var = self.label_var
        return label_var

    def requires(self):
        # first need to find duration of tracked object
        tasks = {}
        kwargs = dict(
            base_name=self.base_name,
            tracking_type=self.tracking_type,
            tracking_timestep_interval=self.tracking_timestep_interval,
            track_without_gal_transform=self.track_without_gal_transform,
        )

        tasks["t_global"] = TrackingVariable2D(var_name="time", **kwargs)
        tasks["agg"] = self._build_agg_tasks()
        return tasks

    def _build_agg_tasks(self):
        times = self.times
        agg_tasks = {}

        for time in times:
            if self.var_name == "z_cb_est":
                # add especialised aggregation task which estimates the cloud-base height
                agg_task = EstimateCloudbaseHeightFromTracking(
                    base_name=self.base_name,
                    time=time,
                    dz=self.dx,
                    track_without_gal_transform=self.track_without_gal_transform,
                    tracking_timestep_interval=self.tracking_timestep_interval,
                )
            else:
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
        agg_output = self.input()["agg"]

        times = self.times

        object_id = self.object_id
        das_agg_obj = []
        # iterate over the timesteps splitting the aggregation in each into
        # different lists for each object present
        for time in tqdm(times, desc="timestep", position=0):
            da_agg_all = agg_output[time].open()
            if "object_id" not in da_agg_all.coords:
                continue

            try:
                da_obj = da_agg_all.sel(object_id=object_id)
            except KeyError:
                # NOTE: the tracking code has as bug so that tmax refers to the next
                # timestep *after* unless the object exists in the very last
                # timestep. We want to know which objects exist at the very last
                # timestep examined, so we need to handle the indexing here and
                # create a fake datasets with nans
                da_obj = np.nan * da_agg_all.isel(object_id=0)
                da_obj = da_obj.assign_coords(dict(object_id=object_id))

            das_agg_obj.append(da_obj)

        # because we're concatenating we need to convert ints to floats,
        # otherwise we'll just get the fill value where there should otherwise
        # be nans
        if np.issubdtype(das_agg_obj[0], np.integer):
            das_agg_obj = [da_.astype(float) for da_ in das_agg_obj]
        da_agg_obj = xr.concat(das_agg_obj, dim="time")

        if self.use_relative_time_axis:
            da_time_relative = da_agg_obj.time - da_agg_obj.time.isel(time=0)
            da_time_relative_mins = (
                da_time_relative.dt.seconds / 60.0
                + 24.0 * 60.0 * da_time_relative.dt.days
            )
            da_time_relative_mins.attrs["long_name"] = "time since forming"
            da_time_relative_mins.attrs["units"] = "min"
            da_agg_obj = da_agg_obj.assign_coords(
                dict(time_relative=da_time_relative_mins)
            ).swap_dims(dict(time="time_relative"))

        da_agg_obj.to_netcdf(self.output().fn)

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
            f"obj_{self.object_id}",
            interval_id,
        ]

        if self.op not in [None, "None"]:
            name_parts.insert(-2, self.op + ["", f"__{str(self.dx)}"][self.dx != None])

        if not self.use_relative_time_axis:
            name_parts.append("absolute_time")

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
    label_var = luigi.OptionalParameter(default=None)
    var_name = luigi.Parameter()
    op = luigi.OptionalParameter(default=None)
    dx = luigi.FloatParameter(default=-1.0)
    use_relative_time_axis = luigi.BoolParameter(default=True)

    track_without_gal_transform = luigi.BoolParameter(default=False)
    tracking_type = luigi.EnumParameter(enum=TrackingType)
    tracking_timestep_interval = luigi.ListParameter([])
    object_ids = luigi.ListParameter(default=[])

    @property
    def _label_var(self):
        if self.var_name == "z_cb_est":
            if self.label_var is [None, "None"]:
                raise Exception(
                    "Shouldn't provide `label_var` when estimating cloud-base height"
                    f" - provided {self.label_var}"
                )
            label_var = "cloud"
        else:
            label_var = self.label_var
        return label_var

    def requires(self):
        # first need to find duration of tracked object
        tasks = {}
        kwargs = dict(
            base_name=self.base_name,
            tracking_type=self.tracking_type,
            tracking_timestep_interval=self.tracking_timestep_interval,
            track_without_gal_transform=self.track_without_gal_transform,
        )

        tasks["t_global"] = TrackingVariable2D(var_name="time", **kwargs)
        tasks["t_start"] = TrackingVariable2D(
            var_name=f"sm{self._label_var}tmin", **kwargs
        )
        tasks["t_end"] = TrackingVariable2D(
            var_name=f"sm{self._label_var}tmax", **kwargs
        )
        return tasks

    @property
    def da_tstart(self):
        if not hasattr(self, "_da_tstart"):
            self._da_tstart = self.input()["t_start"].open()
        return self._da_tstart

    @property
    def da_tend(self):
        if not hasattr(self, "_da_tend"):
            self._da_tend = self.input()["t_end"].open()
        return self._da_tend

    @property
    def da_time(self):
        if not hasattr(self, "_da_time"):
            self._da_time = self.input()["t_global"].open()
        return self._da_time

    def make_object_agg_task(self, object_id):
        obj_var = f"sm{self._label_var}id"

        def _get_times(object_id):
            tstart_obj = self.da_tstart.sel({obj_var: object_id})
            tend_obj = self.da_tend.sel({obj_var: object_id})
            times = self.da_time.sel(time=slice(tstart_obj, tend_obj)).values

            # super hacky way of sending list of datetimes as luigi parameter
            times = times.astype("datetime64[s]").astype(datetime.datetime).tolist()
            times = [t.isoformat() for t in times]
            return times

        object_times = _get_times(object_id=object_id)

        return Object2DCrossSectionAggregation(
            var_name=self.var_name,
            base_name=self.base_name,
            dx=self.dx,
            op=self.op,
            label_var=self.label_var,
            object_id=object_id,
            times=object_times,
            tracking_timestep_interval=self.tracking_timestep_interval,
            tracking_type=self.tracking_type,
            track_without_gal_transform=self.track_without_gal_transform,
        )

    def run(self):
        inputs = self.input()
        obj_var = f"sm{self._label_var}id"

        if len(self.object_ids) == 0:
            da_tstart = inputs["t_start"].open()
            object_ids = da_tstart[obj_var].astype(int).values
        else:
            object_ids = self.object_ids

        agg_tasks = {}
        agg_output_all = {}

        for object_id in tqdm(object_ids, desc="objects", position=0):
            agg_tasks[object_id] = self.make_object_agg_task(object_id=object_id)

            if len(agg_tasks) > N_parallel_tasks:
                agg_tasks_output = yield agg_tasks
                agg_output_all.update(agg_tasks_output)
                agg_tasks = {}

        agg_tasks_output = yield agg_tasks
        agg_output_all.update(agg_tasks_output)

        das_agg = []
        for object_id in tqdm(object_ids, desc="object"):
            da_agg_obj = agg_output_all[object_id].open()
            das_agg.append(da_agg_obj)

        da = xr.concat(das_agg, dim="object_id")

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

        if self.op is not None:
            name_parts.insert(-2, self.op + ["", f"__{str(self.dx)}"][self.dx != None])

        if not self.use_relative_time_axis:
            name_parts.append("absolute_time")

        if self.track_without_gal_transform:
            name_parts.append("go_track")

        if self.object_ids:
            ids_string = ":".join(["{:d}".format(o_id) for o_id in self.object_ids])
            ids_hash = md5(ids_string.encode("ascii")).hexdigest()
            name_parts.append(ids_hash)

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

        return ExtractField3D(
            base_name=self.base_name,
            field_name=self.ref_field_name,
        )

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


class EstimateCloudbaseHeightFromTracking(luigi.Task):
    """
    From a histogram of the cloud-underside heights for all tracked clouds
    present at a given time estimate the cloud-base height as the peak of this
    histogram
    """

    base_name = luigi.Parameter()
    time = NumpyDatetimeParameter()
    dz = luigi.FloatParameter()
    tracking_timestep_interval = luigi.ListParameter(default=[])
    track_without_gal_transform = luigi.BoolParameter(default=False)

    def requires(self):
        reqs = {}
        reqs["cldbase_binned"] = Aggregate2DCrossSectionOnTrackedObjects(
            var_name="cldbase",
            base_name=self.base_name,
            time=self.time,
            label_var="cloud",
            tracking_type=TrackingType.CLOUD_CORE,
            op="histogram",
            dx=self.dz,
            tracking_timestep_interval=self.tracking_timestep_interval,
            track_without_gal_transform=self.track_without_gal_transform,
        )
        reqs["cldbase"] = TimeCrossSectionSlices2D(
            base_name=self.base_name, var_name="cldbase"
        )
        return reqs

    def run(self):
        da_cldbase_histogram = self.input()["cldbase_binned"].open()
        if da_cldbase_histogram.dropna(dim="time").count() == 0:
            da_zb = xr.DataArray()
        elif da_cldbase_histogram.object_id.count() == 0:
            da_zb = xr.DataArray()
        else:
            da_zb = cloudbase_max_height_by_histogram_peak(
                da_cldbase_histogram=da_cldbase_histogram
            )

        da_zb.name = "z_cb_est"
        da_zb.to_netcdf(self.output().fn)

    def output(self):
        name_parts = [
            "z_cb_from_hist_peak",
            str(self.time).replace("-", "").replace(":", "").replace(" ", ""),
            f"dz_{self.dz}",
        ]

        if self.track_without_gal_transform:
            name_parts.append("go_track")

        if self.tracking_timestep_interval:
            interval_id = "tn{}_to_tn{}".format(*self.tracking_timestep_interval)
        else:
            interval_id = "__all__"

        name_parts.append(interval_id)

        fn = "{}.nc".format(".".join(name_parts))
        p = get_workdir() / self.base_name / "cross_sections" / "aggregated" / fn
        return XArrayTarget(str(p))


class TrackedObjectFirstRiseExtraction(luigi.Task):
    base_name = luigi.Parameter()
    label_var = luigi.Parameter(default="cloud")
    var_name = luigi.Parameter(default="cldtop")

    track_without_gal_transform = luigi.BoolParameter(default=False)
    tracking_timestep_interval = luigi.ListParameter([])
    object_ids = luigi.ListParameter(default=[])

    def requires(self):
        task_ztop = AllObjectsAll2DCrossSectionAggregations(
            base_name=self.base_name,
            label_var=self.label_var,
            var_name=self.var_name,
            op="maximum",
            track_without_gal_transform=self.track_without_gal_transform,
            tracking_type=self.tracking_type,
            tracking_timestep_interval=self.tracking_timestep_interval,
            object_ids=self.object_ids,
        )

        task_cloudtype = TrackingVariable2D(
            base_name=self.base_name,
            var_name=f"sm{self.label_var}type",
            tracking_type=self.tracking_type,
            track_without_gal_transform=self.track_without_gal_transform,
            tracking_timestep_interval=self.tracking_timestep_interval,
        )

        return dict(ztop=task_ztop, cloudtype=task_cloudtype)

    def run(self):
        inputs = self.input()

        da_cloudtype = inputs["cloudtype"].open()
        da_z = inputs["ztop"].open()

        da_cloudtype = da_cloudtype.rename(smcloudid="object_id").astype(int)
        da_cloudtype["object_id"] = da_cloudtype.coords["object_id"].astype(int)

        def is_peak(da):
            values = da.values
            peaks = np.zeros(da.shape)
            is_incr = values[:-1] < values[1:]
            left_incr = is_incr[:-1]
            right_incr = is_incr[1:]
            peaks[1:-1] = np.logical_and(left_incr, ~right_incr)
            return xr.DataArray(peaks, dims=da.dims, coords=da.coords)

        def extract_to_peak(da_z, dim_time="time_relative", t_extra=0):
            mask_peaks = is_peak(da_z)
            peaks = da_z.where(mask_peaks, drop=True)
            if peaks.count() > 0:
                t_peak = peaks.isel(**{dim_time: 0})[dim_time] + t_extra
                da_z_to_peak = da_z.sel(**{dim_time: slice(None, t_peak)})
                # if len(peaks) > 1:
                # t_peak2 = peaks.isel(**{dim_time: 1})[dim_time] + t_extra
                # da_z_1peak2 = da_z.sel(**{dim_time: slice(t_peak, t_peak2)})

                # if da_z_1peak2.min() + 100 < da_z_to_peak.max()
                # return np.nan * xr.zeros_like(da_z)

                if t_peak > 5.0:
                    return da_z_to_peak
                else:
                    return np.nan * xr.zeros_like(da_z)
            else:
                return np.nan * xr.zeros_like(da_z)

        def duration(da):
            return xr.DataArray(int(da.where(da > 0, drop=True).count()))

        def unique_values(da):
            return xr.DataArray(len(np.unique(da.values)))

        def z_start(da):
            return xr.DataArray(da.isel(time_relative=0).item())

        # da_z_singlecore = da_z.where(da_cloudtype == CloudType.SINGLE_PULSE, drop=True)
        # da_ = da_z_singlecore

        da_ = da_z

        da_["z_start"] = da_.groupby("object_id").apply(z_start)
        da_["duration"] = da_.groupby("object_id").apply(duration)
        da_["unique_values"] = da_.groupby("object_id").apply(unique_values)

        da_ = da_.where(
            np.logical_and(
                np.logical_and(
                    # filter to only include objects that last a minimum duration
                    da_.duration > 3,
                    # exclude objects which just have a single repeated z-value and then nothing
                    da_.unique_values > 3,
                ),
                # include only objects which start with z_max below 800m
                # (otherwise we haven't tracked these clouds for very long)
                da_.z_start < 1400.0,
            ),
            drop=True,
        )

        da_to_peak = da_.groupby("object_id").apply(extract_to_peak, t_extra=0)
        da_to_peak = da_to_peak.where(da_to_peak)
        da_to_peak.to_netcdf(self.output().fn)

    @property
    def tracking_type(self):
        if self.label_var == "cloud":
            tracking_type = TrackingType.CLOUD_CORE
        else:
            raise NotImplementedError(self.label_var)
        return tracking_type

    def output(self):
        type_id = uclales_2d_tracking.TrackingType.make_identifier(self.tracking_type)
        if self.tracking_timestep_interval:
            interval_id = "tn{}_to_tn{}".format(*self.tracking_timestep_interval)
        else:
            interval_id = "__all__"

        name_parts = [
            self.var_name,
            f"of_{self.label_var}",
            "first_rise",
            f"tracked_{type_id}",
            interval_id,
        ]

        if self.track_without_gal_transform:
            name_parts.append("go_track")

        fn = f"{'.'.join(name_parts)}.nc"
        p = get_workdir() / self.base_name / "cross_sections" / "aggregated" / fn
        return XArrayTarget(str(p))
