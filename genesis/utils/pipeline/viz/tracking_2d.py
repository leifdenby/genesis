import luigi
import matplotlib.pyplot as plt
import numpy as np
import datetime

from .. import data


class CloudCrossSectionAnimationFrame(luigi.Task):
    base_name = luigi.Parameter(default=None)
    time = data.base.NumpyDatetimeParameter()
    center_pt = luigi.ListParameter(default=[])
    l_pad = luigi.FloatParameter(default=5000)
    remove_gal_transform = luigi.BoolParameter(default=False)
    scalar = luigi.Parameter(default="lwp")
    label_var = luigi.Parameter(default="nrcloud")

    def requires(self):
        if self.label_var == "nrthrm":
            tracking_type = data.tracking_2d.TrackingType.CLOUD_CORE_THERMAL
        else:
            tracking_type = data.tracking_2d.TrackingType.CLOUD_CORE

        tasks = dict(
            labels=data.tracking_2d.TrackingLabels2D(
                base_name=self.base_name,
                tracking_type=tracking_type,
                remove_gal_transform=self.remove_gal_transform,
                label_var=self.label_var,
                time=self.time,
            ),
            scalar=data.extraction.ExtractCrossSection2D(
                base_name=self.base_name,
                field_name=self.scalar,
                remove_gal_transform=self.remove_gal_transform,
                time=self.time,
            ),
        )

        object_type = self._get_object_type()
        for grid_var in ["xt", "yt"]:
            v = "{}_{}".format(grid_var[0], object_type)
            field_name = grid_var

            tasks[v] = data.tracking_2d.Aggregate2DCrossSectionOnTrackedObjects(
                base_name=self.base_name,
                field_name=field_name,
                op="mean",
                label_var=self.label_var,
                time=self.time,
                remove_gal_transform=self.remove_gal_transform,
                tracking_type=tracking_type,
            )

        return tasks

    def _get_object_type(self):
        if self.label_var == "nrcloud":
            object_type = "cloud"
        elif self.label_var == "nrthrm":
            object_type = "thermal"
        else:
            raise NotImplementedError(self.label_var)
        return object_type

    def run(self):
        if len(self.center_pt) == 2:
            x_c, y_c = self.center_pt
            l_pad = self.l_pad

            kws = dict(
                xt=slice(x_c - l_pad / 2.0, x_c + l_pad / 2.0),
                yt=slice(y_c - l_pad / 2.0, y_c + l_pad / 2.0),
            )
        else:
            kws = {}

        fig, ax = plt.subplots(figsize=(16, 12))

        da_scalar = self.input()["scalar"].open()

        da_labels = self.input()["labels"].open()

        object_type = self._get_object_type()
        da_x_object = self.input()[f"x_{object_type}"].open()
        da_y_object = self.input()[f"y_{object_type}"].open()

        (
            da_scalar
            .sel(**kws)
            .plot(ax=ax, vmax=0.1, add_colorbar=True, cmap="Blues")
        )
        (
            da_labels.astype(int)
            .sel(**kws)
            .plot.contour(ax=ax, colors=["red"], levels=[0.5])
        )

        # plot object centers computed by aggregating over object label masks
        text_bbox = dict(facecolor="white", alpha=0.5, edgecolor="none")
        object_ids = np.unique(da_labels.sel(**kws))
        object_ids = object_ids[~np.isnan(object_ids)]
        for c_id in object_ids:
            x_t = da_x_object.sel(object_id=c_id)
            y_t = da_y_object.sel(object_id=c_id)

            ax.scatter(x_t, y_t, marker="x", color="red")
            ax.text(x_t, y_t, int(c_id), color="red", bbox=text_bbox)

        if len(self.center_pt) == 2:
            x_c, y_c = self.center_pt
            ax.axhline(y=y_c, linestyle="--", color="grey")
            ax.axvline(x=x_c, linestyle="--", color="grey")

        ax.set_aspect(1)
        plt.savefig(str(self.output().fn))

    def output(self):
        fn = "{}.{}__frame.{}.{}_gal_transform.{}.png".format(
            self.base_name, self.label_var, self.scalar,
            ["with", "without"][self.remove_gal_transform], self.time.isoformat().replace(":", "")
        )
        return luigi.LocalTarget(fn)


class CloudCrossSectionAnimationSpan(CloudCrossSectionAnimationFrame):
    t_start_offset_hrs = luigi.IntParameter()
    t_duration_mins = luigi.IntParameter()

    def requires(self):
        return data.extraction.TimeCrossSectionSlices2D(
            base_name=self.base_name,
            field_name=self.scalar,
        )

    def _build_subtasks(self):
        da_scalar_2d = self.input().open()
        t0 = da_scalar_2d.time.min()
        t_start = t0 + np.timedelta64(self.t_start_offset_hrs, "h")
        t_end = t_start + np.timedelta64(self.t_duration_mins, "m")

        da_times = da_scalar_2d.sel(time=slice(t_start, t_end)).time

        tasks = [
            CloudCrossSectionAnimationFrame(
                base_name=self.base_name,
                time=t,
                center_pt=self.center_pt,
                l_pad=self.l_pad,
                remove_gal_transform=self.remove_gal_transform,
                scalar=self.scalar,
                label_var=self.label_var
            )
            for t in da_times
        ]
        return tasks

    def run(self):
        yield self._build_subtasks()

    def output(self):
        return [
            t.output() for t in self._build_subtasks()
        ]
