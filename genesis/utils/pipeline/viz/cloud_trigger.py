import luigi
import matplotlib.pyplot as plt

from .. import data


class CloudTriggeringCrossSectionAnimationFrame(luigi.Task):
    base_name = luigi.Parameter(default=None)
    time = data.base.NumpyDatetimeParameter()
    center_pt = luigi.ListParameter(default=[])
    l_pad = luigi.FloatParameter(default=5000)
    remove_gal_transform = luigi.BoolParameter(default=False)

    def requires(self):
        return dict(
            nrcloud=data.tracking_2d.TrackingLabels2D(
                base_name=self.base_name,
                tracking_type=data.tracking_2d.TrackingType.CLOUD_CORE_THERMAL,
                remove_gal_transform=self.remove_gal_transform,
                label_var='nrcloud',
                time=self.time,
            ),
            nrthrm=data.tracking_2d.TrackingLabels2D(
                base_name=self.base_name,
                tracking_type=data.tracking_2d.TrackingType.CLOUD_CORE_THERMAL,
                remove_gal_transform=self.remove_gal_transform,
                label_var='nrthrm',
                time=self.time,
            ),
            trctop=data.extraction.ExtractCrossSection2D(
                base_name=self.base_name, field_name="trctop", remove_gal_transform=self.remove_gal_transform,
                time=self.time
            ),
            lwp=data.extraction.ExtractCrossSection2D(
                base_name=self.base_name, field_name="lwp", remove_gal_transform=self.remove_gal_transform,
                time=self.time
            ),
        )

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

        fig, ax = plt.subplots(figsize=(8, 6))

        da_trctop = self.input()["trctop"].open()
        da_lwp = self.input()["lwp"].open()

        da_nrcloud = self.input()['nrcloud'].open()
        da_nrthrm = self.input()['nrthrm'].open()

        (
            da_trctop.where(da_trctop > 0)
            .sel(**kws)
            .plot(ax=ax, add_colorbar=True, cmap="Reds", vmax=2000)
        )
        (
            da_lwp.where(da_lwp > 0)
            .sel(**kws)
            .plot(ax=ax, vmax=0.1, add_colorbar=True, cmap="Blues")
        )
        (
            da_nrthrm.astype(int)
            .sel(**kws)
            .plot.contour(ax=ax, color='black', levels=[0.5,])
        )

        # import ipdb
        # with ipdb.launch_ipdb_on_exception():

            # therm_ids = np.unique(da_nrthrm)
            # therm_ids = therm_ids[~np.isnan(therm_ids)]

            # for t_id in therm_ids:
                # x_t = ds_tracking.sel(smthrmid=t_id).smthrmx
                # y_t = ds_tracking.sel(smthrmid=t_id).smthrmy

                # ax.scatter(x_t, y_t, marker='x', color='black')
                # # ax.text(x_t, y_t, int(t_id))

        if len(self.center_pt) == 2:
            x_c, y_c = self.center_pt
            ax.axhline(y=y_c, linestyle="--", color="grey")
            ax.axvline(x=x_c, linestyle="--", color="grey")

        ax.set_aspect(1)
        plt.savefig(str(self.output().fn))

    def output(self):
        fn = "cloud_trigger__frame__{}.png".format(self.time.isoformat())
        return luigi.LocalTarget(fn)
