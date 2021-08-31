from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import xarray as xr

from .....bulk_statistics import cross_correlation_with_height
from ...data.base import XArrayTarget
from ...data.extraction import ExtractField3D
from ...data.masking import MakeMask
from ...data.tracking_2d.cloud_base import ExtractBelowCloudEnvironment


class JointDistProfile(luigi.Task):
    dk = luigi.IntParameter()
    z_max = luigi.FloatParameter(significant=False, default=700.0)
    v1 = luigi.Parameter()
    v2 = luigi.Parameter()
    base_name = luigi.Parameter()

    mask_method = luigi.Parameter(default=None)
    mask_method_extra_args = luigi.Parameter(default="")
    plot_limits = luigi.ListParameter(default=None)
    data_only = luigi.BoolParameter(default=False)
    cloud_age_max = luigi.FloatParameter(default=200.0)
    cumulative_contours = luigi.Parameter(default="10,90")
    add_mean_ref = luigi.BoolParameter(default=False)
    add_cloudbase_peak_ref = luigi.Parameter(default=False)
    add_legend = luigi.Parameter(default=True)
    figsize = luigi.ListParameter(default=[4.0, 3.0])

    def requires(self):
        reqs = dict(
            full_domain=[
                ExtractField3D(field_name=self.v1, base_name=self.base_name),
                ExtractField3D(field_name=self.v2, base_name=self.base_name),
            ],
        )

        if self.add_cloudbase_peak_ref:
            reqs["cloudbase"] = [
                ExtractBelowCloudEnvironment(
                    base_name=self.base_name,
                    field_name=self.v1,
                    cloud_age_max=self.cloud_age_max,
                    ensure_tracked=self.add_cloudbase_peak_ref == "tracked_only",
                ),
                ExtractBelowCloudEnvironment(
                    base_name=self.base_name,
                    field_name=self.v2,
                    cloud_age_max=self.cloud_age_max,
                    ensure_tracked=self.add_cloudbase_peak_ref == "tracked_only",
                ),
            ]

        if self.mask_method is not None:
            reqs["mask"] = MakeMask(
                method_name=self.mask_method,
                method_extra_args=self.mask_method_extra_args,
                base_name=self.base_name,
            )

        return reqs

    def output(self):
        if self.mask_method is not None:
            mask_name = MakeMask.make_mask_name(
                base_name=self.base_name,
                method_name=self.mask_method,
                method_extra_args=self.mask_method_extra_args,
            )
            out_fn = "{}.cross_correlation.{}.{}.masked_by.{}.png".format(
                self.base_name, self.v1, self.v2, mask_name
            )
        else:
            out_fn = "{}.cross_correlation.{}.{}.png".format(
                self.base_name, self.v1, self.v2
            )

        if self.data_only:
            out_fn = out_fn.replace(".png", ".nc")
            p_out = Path("data") / self.base_name / out_fn
            return XArrayTarget(str(p_out))
        else:
            out_fn = out_fn.replace(
                ".png",
                ".{}__contour_levels.png".format(
                    self.cumulative_contours.replace(",", "__")
                ),
            )
            return luigi.LocalTarget(out_fn)

    def run(self):
        ds_3d = xr.merge([xr.open_dataarray(r.fn) for r in self.input()["full_domain"]])
        if "cloudbase" in self.input():
            ds_cb = xr.merge(
                [xr.open_dataarray(r.fn) for r in self.input()["cloudbase"]]
            )
        else:
            ds_cb = None

        mask = None
        if "mask" in self.input():
            mask = self.input()["mask"].open()
            ds_3d = ds_3d.where(mask)

        ds_3d = ds_3d.sel(zt=slice(0, self.z_max))

        ds_3d_levels = ds_3d.isel(zt=slice(None, None, self.dk))

        if self.data_only:
            if "mask" in self.input():
                ds_3d_levels.attrs["mask_desc"] = mask.long_name
            ds_3d_levels.to_netcdf(self.output().fn)
        else:
            self.make_plot(
                ds_3d=ds_3d, ds_cb=ds_cb, ds_3d_levels=ds_3d_levels, mask=mask
            )
            plt.savefig(self.output().fn, bbox_inches="tight")

    def make_plot(self, ds_3d, ds_cb, ds_3d_levels, mask):
        fig_w, fig_h = self.figsize

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        if self.add_mean_ref:
            ax.axvline(x=ds_3d[self.v1].mean(), color="grey", alpha=0.4)
            ax.axhline(y=ds_3d[self.v2].mean(), color="grey", alpha=0.4)

        normed_levels = [int(v) for v in self.cumulative_contours.split(",")]
        ax, _ = cross_correlation_with_height.main(
            ds_3d=ds_3d_levels,
            ds_cb=ds_cb,
            normed_levels=normed_levels,
            ax=ax,
            add_cb_peak_ref_line=self.add_cloudbase_peak_ref,
            add_legend=self.add_legend,
        )

        title = ax.get_title()
        title = "{}\n{}".format(self.base_name, title)
        if "mask" in self.input():
            title += "\nmasked by {}".format(mask.long_name)
        ax.set_title(title)

        if self.plot_limits:
            x_min, x_max, y_min, y_max = self.plot_limits

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        return ax
