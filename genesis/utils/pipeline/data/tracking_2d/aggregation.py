from pathlib import Path
import os


import luigi
import xarray as xr
import numpy as np
import dask_image.ndmeasure as dmeasure
from tqdm import tqdm


from .....utils import find_vertical_grid_spacing, find_horizontal_grid_spacing
from ...data_sources import uclales_2d_tracking
from ..base import (
    get_workdir,
    _get_dataset_meta_info,
    XArrayTarget,
    NumpyDatetimeParameter,
)
from ..extraction import (
    ExtractCrossSection2D,
    ExtractField3D,
    REGEX_INSTANTENOUS_BASENAME,
    remove_gal_transform,
)
from .base import TrackingVariable2D, TrackingLabels2D
from . import TrackingType


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
        elif self.op in vars(dmeasure):
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
            meta = _get_dataset_meta_info(self.base_name)
            u_gal, v_gal = meta["U_gal"]
            name_parts.append(f"go_labels_{u_gal}_{v_gal}")

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
    use_relative_time_axis = luigi.BoolParameter(default=True)

    track_without_gal_transform = luigi.BoolParameter(default=False)
    tracking_type = luigi.EnumParameter(enum=TrackingType)
    tracking_timestep_interval = luigi.ListParameter([])
    timestep_skip = luigi.IntParameter(default=None)

    def requires(self):
        # first need to find duration of tracked object
        tasks = {}
        kwargs = dict(
            base_name=self.base_name,
            tracking_type=self.tracking_type,
            tracking_timestep_interval=self.tracking_timestep_interval,
            track_without_gal_transform=self.track_without_gal_transform,
        )

        tasks["t_start"] = TrackingVariable2D(
            var_name=f"sm{self.label_var}tmin", **kwargs
        )
        tasks["t_end"] = TrackingVariable2D(
            var_name=f"sm{self.label_var}tmax", **kwargs
        )
        tasks["t_global"] = TrackingVariable2D(var_name="time", **kwargs)
        return tasks

    def _build_agg_tasks(self):
        times = self._get_times()
        agg_tasks = {}

        for time in times:
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

    def _get_times(self):
        da_time = self.input()["t_global"].open()
        times = da_time.values
        if self.timestep_skip is not None:
            times = times[:: self.timestep_skip]
        return times

    def run(self):
        agg_tasks = self._build_agg_tasks()
        agg_output = yield agg_tasks

        inputs = self.input()
        times = self._get_times()
        da_tstart = inputs["t_start"].open()
        da_tend = inputs["t_end"].open()
        # dt = np.gradient(da_time.values)[0]

        obj_var = f"sm{self.label_var}id"
        all_object_ids = da_tstart[obj_var].astype(int)

        open_aggs = {}
        das_agg_objs = {}
        # iterate over the timesteps splitting the aggregation in each into
        # different lists for each object present
        for time in tqdm(times, desc="timestep", position=0):
            da_agg_all = agg_output[time].open()
            if "object_id" not in da_agg_all.coords:
                continue

            # find all objects that were present at this time

            object_ids = da_agg_all.object_id
            # XXX: it appears there may be another bug here, this time with
            # the start/end time of object appearance
            # object_ids = all_object_ids.where(
            # ((da_tstart <= time) * (time <= da_tend)), drop=True
            # )

            for object_id in object_ids.values:
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

            # XXX: appears there's abug with the cloud tracking code... so I've
            # disabled this for now
            # t_start_obj = da_tstart.sel({obj_var: object_id})
            # assert t_start_obj == da_agg_obj.time.isel(time=0)
            # t_end_obj = da_tend.sel({obj_var: object_id})
            # if not t_end_obj == da_agg_obj.time.isel(time=-1):
            # # NOTE: again the bug in the cloud-tracking code means that
            # # objects that exist at the end of the time-range have their
            # # end time actually *after* the very last timestep
            # if t_end_obj == da_time.isel(time=-1) + dt:
            # pass
            # else:
            # raise Exception("")

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
            self.op + ["", f"__{str(self.dx)}"][self.dx != None],
            interval_id,
        ]

        if not self.use_relative_time_axis:
            name_parts.append("absolute_time")

        if self.timestep_skip is not None:
            name_parts.append(f"{self.timestep_skip}tn_skip")

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
