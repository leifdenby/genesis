import datetime

import luigi
import matplotlib.patches as mpl_patches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from skimage.color import colorlabel

from .. import data
from .all import _scale_dist

label2rgb = colorlabel.label2rgb


class FixedColorMap:
    def _match_label_with_color(self, label, colors, bg_label, bg_color):
        mm, color_cycle = self._old_match_label_with_color(
            label=label,
            colors=colors,
            bg_label=bg_label,
            bg_color=bg_color,
        )
        mapped_labels = label.flatten() % len(colors)
        return mapped_labels, color_cycle

    def __enter__(self):
        self._old_match_label_with_color = colorlabel._match_label_with_color
        colorlabel._match_label_with_color = self._match_label_with_color

    def __exit__(self, type, value, traceback):
        colorlabel._match_label_with_color = self._old_match_label_with_color


class CloudCrossSectionAnimationFrame(luigi.Task):
    base_name = luigi.Parameter(default=None)
    time = data.base.NumpyDatetimeParameter()
    center_pt = luigi.ListParameter(default=[])
    l_pad = luigi.FloatParameter(default=5000)
    track_without_gal_transform = luigi.BoolParameter(default=False)
    remove_gal_transform = luigi.BoolParameter(default=True)
    var_name = luigi.Parameter(default="lwp")
    label_var = luigi.Parameter(default="cloud")
    coloured_labels = luigi.BoolParameter(default=False)
    tracking_timestep_interval = luigi.ListParameter(default=[])
    scalar_cmap = luigi.Parameter(default="Blues")
    label_annotation = luigi.ChoiceParameter(
        choices=["", "bounding_box", "object_id"], var_type=str, default=""
    )
    figsize = luigi.ListParameter(default=(16, 12))
    no_title = luigi.BoolParameter(default=False)

    def requires(self):
        if self.label_var == "thrm":
            tracking_type = data.tracking_2d.TrackingType.CLOUD_CORE_THERMAL
        else:
            tracking_type = data.tracking_2d.TrackingType.CLOUD_CORE

        tasks = dict(
            labels=data.tracking_2d.TrackingLabels2D(
                base_name=self.base_name,
                tracking_type=tracking_type,
                track_without_gal_transform=self.track_without_gal_transform,
                label_var=self.label_var,
                time=self.time,
                tracking_timestep_interval=self.tracking_timestep_interval,
                offset_labels_by_gal_transform=self.remove_gal_transform,
            ),
            scalar=data.extraction.ExtractCrossSection2D(
                base_name=self.base_name,
                var_name=self.var_name,
                remove_gal_transform=self.remove_gal_transform,
                time=self.time,
            ),
        )

        object_type = self._get_object_type()
        for grid_var in ["xt", "yt"]:
            ops = [
                "mean",
            ]
            if self.label_annotation == "bounding_box":
                ops += ["minimum", "maximum"]

            for op in ops:
                v = f"{grid_var[0]}_{object_type}_{op}"
                var_name = grid_var

                tasks[v] = data.tracking_2d.Aggregate2DCrossSectionOnTrackedObjects(
                    base_name=self.base_name,
                    var_name=var_name,
                    op=op,
                    label_var=self.label_var,
                    time=self.time,
                    track_without_gal_transform=self.track_without_gal_transform,
                    tracking_type=tracking_type,
                    offset_labels_by_gal_transform=self.remove_gal_transform,
                    tracking_timestep_interval=self.tracking_timestep_interval,
                )

        return tasks

    def _get_object_type(self):
        if self.label_var.startswith("cloud"):
            object_type = self.label_var
        elif self.label_var == "thrm":
            object_type = "thermal"
        elif self.label_var == "cldthrm_family":
            object_type = "cldthrm_family"
        else:
            raise NotImplementedError(self.label_var)
        return object_type

    def _make_plot(self, ax=None):
        if len(self.center_pt) == 2:
            x_c, y_c = self.center_pt
            l_pad = self.l_pad

            kws = dict(
                xt=slice(x_c - l_pad / 2.0, x_c + l_pad / 2.0),
                yt=slice(y_c - l_pad / 2.0, y_c + l_pad / 2.0),
            )
        else:
            kws = {}

        da_scalar = self.input()["scalar"].open()
        da_labels = self.input()["labels"].open()

        da_labels = da_labels.assign_coords(
            xt=_scale_dist(da_labels.xt), yt=_scale_dist(da_labels.yt)
        )
        da_scalar = da_scalar.assign_coords(
            xt=_scale_dist(da_scalar.xt), yt=_scale_dist(da_scalar.yt)
        )

        object_type = self._get_object_type()
        da_x_object = self.input()[f"x_{object_type}_mean"].open()
        da_y_object = self.input()[f"y_{object_type}_mean"].open()

        (
            da_scalar.sel(**kws).plot(
                ax=ax,
                add_colorbar=True,
                cmap=self.scalar_cmap,
                zorder=1,
                robust=True,
            )
        )

        if self.coloured_labels:
            da_ = da_labels.sel(**kws).fillna(0).astype(int)
            with FixedColorMap():
                rgb_values = label2rgb(
                    da_.values, alpha=0.2, bg_label=0, bg_color=(255, 255, 255)
                )
            rgba_values = np.zeros(
                (lambda nx, ny, nc: (nx, ny, nc + 1))(*rgb_values.shape)
            )
            rgba_values[:, :, :-1] = rgb_values
            rgba_values[:, :, -1] = 0.2
            da_rgb = xr.DataArray(
                rgb_values, coords=da_.coords, dims=list(da_.dims) + ["rgb"]
            )
            da_rgb.plot.imshow(ax=ax, rgb="rgb", alpha=0.5, zorder=10)
        else:
            (
                da_labels.astype(int)
                .sel(**kws)
                .plot.contour(ax=ax, colors=["black"], levels=[0.5])
            )

        if self.label_annotation == "bounding_box":
            da_xmin_object = self.input()[f"x_{object_type}_minimum"].open()
            da_xmax_object = self.input()[f"x_{object_type}_maximum"].open()
            da_ymin_object = self.input()[f"y_{object_type}_minimum"].open()
            da_ymax_object = self.input()[f"y_{object_type}_maximum"].open()

        # plot object centers computed by aggregating over object label masks
        text_bbox = dict(facecolor="white", alpha=0.5, edgecolor="none")
        object_ids = np.unique(da_labels.sel(**kws))
        object_ids = object_ids[~np.isnan(object_ids)]
        if object_ids[0] == 0:
            object_ids = object_ids[1:]
        for c_id in object_ids:
            x_t = da_x_object.sel(object_id=c_id)
            y_t = da_y_object.sel(object_id=c_id)

            if self.label_annotation in ["bounding_box", "object_id"]:
                ax.scatter(x_t, y_t, marker="x", color="red")

            if self.label_annotation == "bounding_box":
                o_xmin = da_xmin_object.sel(object_id=c_id)
                o_xmax = da_xmax_object.sel(object_id=c_id)
                o_ymin = da_ymin_object.sel(object_id=c_id)
                o_ymax = da_ymax_object.sel(object_id=c_id)
                rect = mpl_patches.Rectangle(
                    (o_xmin, o_ymin),
                    (o_xmax - o_xmin),
                    (o_ymax - o_ymin),
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                    linestyle="--",
                )
                ax.add_patch(rect)
                ax.text(
                    o_xmin,
                    o_ymax,
                    int(c_id),
                    color="red",
                    bbox=text_bbox,
                    horizontalalignment="left",
                    verticalalignment="bottom",
                )
            elif self.label_annotation == "object_id":
                ax.text(x_t, y_t, int(c_id), color="red", bbox=text_bbox)

        if len(self.center_pt) == 2:
            x_c, y_c = self.center_pt
            ax.axhline(y=y_c, linestyle="--", color="grey")
            ax.axvline(x=x_c, linestyle="--", color="grey")

        if not self.no_title:
            title_parts = [
                f"{self.var_name} field with {self.label_var} labels",
                "{} gal offset tracking, {} gal offset labels".format(
                    ["without", "with"][self.track_without_gal_transform],
                    ["without", "with"][self.remove_gal_transform],
                ),
                self.time.isoformat(),
            ]

            ax.set_title("\n".join(title_parts))
        else:
            ax.set_title("")

        ax.set_aspect(1)
        plt.tight_layout()

    def run(self):
        fig, ax = plt.subplots(figsize=self.figsize)
        self._make_plot(ax=ax)
        plt.savefig(str(self.output().fn))

    def output(self):
        name_parts = [
            self.base_name,
            f"{self.label_var}__frame",
            self.var_name,
        ]

        if self.track_without_gal_transform:
            name_parts.append("go_track")
        if self.remove_gal_transform:
            name_parts.append("go_labels")
        if self.coloured_labels:
            name_parts.append("coloured")

        name_parts += [self.time.isoformat().replace(":", ""), "png"]

        fn = ".".join(name_parts)
        return luigi.LocalTarget(fn)


class CloudCrossSectionAnimationSpan(CloudCrossSectionAnimationFrame):
    t_duration_mins = luigi.IntParameter()

    def requires(self):
        return data.extraction.TimeCrossSectionSlices2D(
            base_name=self.base_name,
            var_name=self.var_name,
        )

    def _build_subtasks(self):
        if not self.input().exists():
            return None

        da_scalar_2d = self.input().open()
        t_start = self.time
        t_end = t_start + datetime.timedelta(minutes=self.t_duration_mins)

        # if t_start < da_scalar_2d.isel(time=0).time:
        if da_scalar_2d.sel(time=slice(None, t_start)).time.count() == 0:
            raise Exception(
                f"The start time chosen `{t_start}` is before the"
                f" first time for which the scalar field `{self.var_name}`"
                " is available"
            )

        da_times = da_scalar_2d.sel(time=slice(t_start, t_end)).time

        tasks = [
            CloudCrossSectionAnimationFrame(
                base_name=self.base_name,
                time=t,
                center_pt=self.center_pt,
                l_pad=self.l_pad,
                track_without_gal_transform=self.track_without_gal_transform,
                tracking_timestep_interval=self.tracking_timestep_interval,
                var_name=self.var_name,
                label_var=self.label_var,
                coloured_labels=self.coloured_labels,
                remove_gal_transform=self.remove_gal_transform,
                label_annotation=self.label_annotation,
                scalar_cmap=self.scalar_cmap,
                figsize=self.figsize,
                no_title=self.no_title,
            )
            for t in da_times
        ]
        return tasks

    def run(self):
        yield self._build_subtasks()

    def output(self):
        tasks = self._build_subtasks()
        if tasks is None:
            return luigi.LocalTarget("__invalid_file__")
        else:
            return [t.output() for t in tasks]
