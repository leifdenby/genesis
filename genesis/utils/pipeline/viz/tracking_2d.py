import luigi
import matplotlib.pyplot as plt
import numpy as np

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
        if self.label_var == "nrthermal":
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

        obj_type = "cloud"
        for grid_var in ["xt", "yt"]:
            v = "{}_{}".format(grid_var[0], obj_type)
            field_name = grid_var

            tasks[v] = data.tracking_2d.Aggregate2DCrossSectionOnTrackedObjects(
                base_name=self.base_name,
                field_name=field_name,
                op="mean",
                label_var="nr{}".format(obj_type),
                time=self.time,
                remove_gal_transform=self.remove_gal_transform,
                tracking_type=data.tracking_2d.TrackingType.CLOUD_CORE,
            )

        return tasks

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

        da_x_cloud = self.input()["x_cloud"].open()
        da_y_cloud = self.input()["y_cloud"].open()

        (
            da_scalar
            .sel(**kws)
            .plot(ax=ax, vmax=0.1, add_colorbar=True, cmap="Blues")
        )
        (
            da_labels.astype(int)
            .sel(**kws)
            .plot.contour(ax=ax, colors=["blue"], levels=[0.5])
        )

        import ipdb

        with ipdb.launch_ipdb_on_exception():
            text_bbox = dict(facecolor="white", alpha=0.5, edgecolor="none")

            cloud_ids = np.unique(da_labels.sel(**kws))
            cloud_ids = cloud_ids[~np.isnan(cloud_ids)]
            for c_id in cloud_ids:
                x_t = da_x_cloud.sel(object_id=c_id)
                y_t = da_y_cloud.sel(object_id=c_id)

                ax.scatter(x_t, y_t, marker="x", color="blue")
                ax.text(x_t, y_t, int(c_id), color="blue", bbox=text_bbox)

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
