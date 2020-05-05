import luigi
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from .. import data


class CloudCrossSectionAnimationFrame(luigi.Task):
    base_name = luigi.Parameter(default=None)
    time = data.base.NumpyDatetimeParameter()
    center_pt = luigi.ListParameter(default=[])
    l_pad = luigi.FloatParameter(default=5000)
    remove_gal_transform = luigi.BoolParameter(default=False)

    def requires(self):
        tasks = dict(
            nrcloud=data.tracking_2d.TrackingLabels2D(
                base_name=self.base_name,
                tracking_type=data.tracking_2d.TrackingType.CLOUD_CORE,
                remove_gal_transform=self.remove_gal_transform,
                label_var='nrcloud',
                time=self.time,
            ),
            lwp=data.extraction.ExtractCrossSection2D(
                base_name=self.base_name, field_name="core", remove_gal_transform=self.remove_gal_transform,
                time=self.time
            ),
        )

        obj_type = "cloud"
        for grid_var in ['xt', 'yt']:
            v = "{}_{}".format(grid_var[0], obj_type)
            field_name = grid_var

            tasks[v] = data.tracking_2d.Aggregate2DCrossSectionOnTrackedObjects(
                base_name=self.base_name,
                field_name=field_name,
                op='mean',
                label_var='nr{}'.format(obj_type),
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

        da_lwp = self.input()["lwp"].open()

        da_nrcloud = self.input()['nrcloud'].open()

        da_x_cloud = self.input()['x_cloud'].open()
        da_y_cloud = self.input()['y_cloud'].open()

        (
            da_lwp.where(da_lwp > 0)
            .sel(**kws)
            .plot(ax=ax, vmax=0.1, add_colorbar=True, cmap="Blues")
        )
        (
            da_nrcloud.astype(int)
            .sel(**kws)
            .plot.contour(ax=ax, colors=['blue',], levels=[0.5,])
        )

        import ipdb
        with ipdb.launch_ipdb_on_exception():
            text_bbox = dict(facecolor='white', alpha=0.5, edgecolor='none')

            cloud_ids = np.unique(da_nrcloud.sel(**kws))
            cloud_ids = cloud_ids[~np.isnan(cloud_ids)]
            for c_id in cloud_ids:
                x_t = da_x_cloud.sel(object_id=c_id)
                y_t = da_y_cloud.sel(object_id=c_id)

                ax.scatter(x_t, y_t, marker='x', color='blue')
                ax.text(x_t, y_t, int(c_id), color='blue', bbox=text_bbox)

        if len(self.center_pt) == 2:
            x_c, y_c = self.center_pt
            ax.axhline(y=y_c, linestyle="--", color="grey")
            ax.axvline(x=x_c, linestyle="--", color="grey")

        ax.set_aspect(1)
        plt.savefig(str(self.output().fn))

    def output(self):
        fn = "cloud__frame.{}_gal_transform.{}.png".format(
            ["with", "without"][self.remove_gal_transform],
            self.time.isoformat()
        )
        return luigi.LocalTarget(fn)
