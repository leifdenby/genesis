import matplotlib

matplotlib.use("Agg")  # noqa

from pathlib import Path
import re
import warnings

import luigi
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import shutil

from ....bulk_statistics import cross_correlation_with_height
from .... import objects
from ... import plot_types, figure_metadata

from .. import data

figure_metadata.patch_savefig_for_argv_metadata()


def _scale_dist(da_):
    units = getattr(da_, "units", None)
    if units is None:
        warnings.warn(f"No units given for `{da_.name}`, assuming `m`")
        units = "m"

    if units == "m":
        da_ = da_ / 1000.0
        da_.attrs["units"] = "km"
        da_.attrs["long_name"] = "horz. dist."
    return da_


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
    add_cloudbase_peak_ref = luigi.BoolParameter(default=False)
    add_legend = luigi.BoolParameter(default=True)

    def requires(self):
        reqs = dict(
            full_domain=[
                data.ExtractField3D(field_name=self.v1, base_name=self.base_name),
                data.ExtractField3D(field_name=self.v2, base_name=self.base_name),
            ],
        )

        reqs["cloudbase"] = [
            data.ExtractCloudbaseState(
                base_name=self.base_name,
                field_name=self.v1,
                cloud_age_max=self.cloud_age_max,
            ),
            data.ExtractCloudbaseState(
                base_name=self.base_name,
                field_name=self.v2,
                cloud_age_max=self.cloud_age_max,
            ),
        ]

        if self.mask_method is not None:
            reqs["mask"] = data.MakeMask(
                method_name=self.mask_method,
                method_extra_args=self.mask_method_extra_args,
                base_name=self.base_name,
            )

        return reqs

    def output(self):
        if self.mask_method is not None:
            mask_name = data.MakeMask.make_mask_name(
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
            return data.XArrayTarget(str(p_out))
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
        fig_w, fig_h = 4.0, 3.0

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


def _textwrap(s, l):
    lines = []
    line = ""

    n = 0
    in_tex = False
    for c in s:
        if c == "$":
            in_tex = not in_tex
        if n < l:
            line += c
        else:
            if c == " " and not in_tex:
                lines.append(line)
                line = ""
                n = 0
            else:
                line += c
        n += 1

    lines.append(line)
    return "\n".join(lines)


class JointDistProfileGrid(luigi.Task):
    dk = luigi.IntParameter()
    z_max = luigi.FloatParameter(significant=True, default=700.0)
    v1 = luigi.Parameter()
    v2 = luigi.Parameter()
    base_names = luigi.Parameter()

    separate_axis_limits = luigi.BoolParameter(default=False)
    mask_method = luigi.Parameter(default=None)
    mask_method_extra_args = luigi.Parameter(default="")

    def requires(self):
        reqs = {}

        for base_name in self.base_names.split(","):
            r = dict(
                nomask=JointDistProfile(
                    dk=self.dk,
                    z_max=self.z_max,
                    v1=self.v1,
                    v2=self.v2,
                    base_name=base_name,
                    data_only=True,
                ),
                masked=JointDistProfile(
                    dk=self.dk,
                    z_max=self.z_max,
                    v1=self.v1,
                    v2=self.v2,
                    base_name=base_name,
                    mask_method=self.mask_method,
                    mask_method_extra_args=self.mask_method_extra_args,
                    data_only=True,
                ),
                cloudbase=[
                    data.ExtractCloudbaseState(base_name=base_name, field_name=self.v1),
                    data.ExtractCloudbaseState(base_name=base_name, field_name=self.v2),
                ],
            )
            reqs[base_name] = r

        return reqs

    def run(self):
        base_names = self.base_names.split(",")

        Nx, Ny = 2, len(base_names)
        if self.separate_axis_limits:
            shareaxes = "row"
        else:
            shareaxes = True
        fig, axes = plt.subplots(
            nrows=len(base_names),
            ncols=2,
            sharex=shareaxes,
            sharey=shareaxes,
            figsize=(Nx * 4, Ny * 3 + 2),
        )

        if Ny == 1:
            axes = np.array([axes])

        for i, base_name in enumerate(base_names):
            for j, part in enumerate(["nomask", "masked"]):
                ds_3d = self.input()[base_name][part].open()
                ds_cb = xr.merge(
                    [
                        xr.open_dataarray(r.fn)
                        for r in self.input()[base_name]["cloudbase"]
                    ]
                )

                ax = axes[i, j]

                _, lines = cross_correlation_with_height.main(
                    ds_3d=ds_3d, ds_cb=ds_cb, ax=ax
                )

                if j > 0:
                    ax.set_ylabel("")
                if i < Ny - 1:
                    ax.set_xlabel("")

                title = base_name
                if part == "masked":
                    title += "\nmasked by {}".format(_textwrap(ds_3d.mask_desc, 30))
                ax.set_title(title)

        plt.figlegend(
            handles=lines,
            labels=[l.get_label() for l in lines],
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.0),
        )

        # rediculous hack to make sure matplotlib includes the figlegend in the
        # saved image
        ax = axes[-1, 0]
        ax.text(0.5, -0.2 - Ny * 0.1, " ", transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(self.output().fn, bbox_inches="tight")

    def output(self):
        arr = []
        for base_name in self.base_names.split(","):
            mask_name = data.MakeMask.make_mask_name(
                base_name=base_name,
                method_name=self.mask_method,
                method_extra_args=self.mask_method_extra_args,
            )
            arr.append(mask_name)

        fn_out = "{}.cross_correlation.grid.{}.{}.png".format(
            "__".join(arr), self.v1, self.v2
        )
        return luigi.LocalTarget(fn_out)


class HorizontalMeanProfile(luigi.Task):
    base_name = luigi.Parameter()
    field_names = luigi.Parameter(default="qv,qc,theta_l")
    mask_method = luigi.Parameter(default=None)
    mask_method_extra_args = luigi.Parameter(default="")
    mask_only = luigi.BoolParameter()

    @property
    def _field_names(self):
        return self.field_names.split(",")

    def requires(self):
        reqs = dict(
            fields=[
                data.ExtractField3D(field_name=v, base_name=self.base_name)
                for v in self._field_names
            ]
        )

        if self.mask_method is not None:
            reqs["mask"] = data.MakeMask(
                base_name=self.base_name,
                method_name=self.mask_method,
                method_extra_args=self.mask_method_extra_args,
            )
        return reqs

    def run(self):
        ds = xr.merge([input.open() for input in self.input()["fields"]])

        fig, axes = plt.subplots(ncols=len(self._field_names), sharey=True)
        sns.set(style="ticks")

        mask = None
        if self.mask_method is not None:
            mask = self.input()["mask"].open()

        title = None
        for v, ax in zip(self._field_names, axes):
            da = ds[v]
            if mask is not None:
                if self.mask_only:
                    other = np.nan
                else:
                    other = 0.0
                da = da.where(mask, other=other)
            v_mean = da.mean(dim=("xt", "yt"), dtype=np.float64, keep_attrs=True)
            v_mean.plot(ax=ax, y="zt")
            ax.set_ylim(0, None)
            ax.set_ylabel("")
            title = ax.get_title()
            ax.set_title("")

        sns.despine()
        plt.suptitle("{}\n{}".format(self.base_name, title))

        plt.savefig(self.output().fn)

    def output(self):
        if self.mask_method is not None:
            mask_name = data.MakeMask.make_mask_name(
                base_name=self.base_name,
                method_name=self.mask_method,
                method_extra_args=self.mask_method_extra_args,
            )
            if self.mask_only:
                mask_name += "_only"
        else:
            mask_name = "nomask"

        fn = "{base_name}.{variables}.{mask_name}.mean_profile.png".format(
            base_name=self.base_name,
            variables="__".join(self._field_names),
            mask_name=mask_name,
        )
        return luigi.LocalTarget(fn)


class CrossSection(luigi.Task):
    base_name = luigi.ListParameter()
    field_name = luigi.Parameter()
    z = luigi.ListParameter()
    no_title = luigi.BoolParameter(default=False)

    def requires(self):
        return dict(
            [
                (
                    base_name,
                    data.ExtractField3D(
                        base_name=base_name, field_name=self.field_name
                    ),
                )
                for base_name in np.atleast_1d(self.base_name)
            ]
        )

    def run(self):
        self._make_plot()
        plt.savefig(self.output().fn, bbox_inches="tight")

    def _make_plot(self):
        da_ = []
        base_names = []
        for base_name, input in self.input().items():
            da_bn = input.open()
            da_bn["base_name"] = base_name
            da_.append(da_bn)
            base_names.append(base_name)

        da = xr.concat(da_, dim="base_name")

        da = da.assign_coords(xt=_scale_dist(da.xt), yt=_scale_dist(da.yt))

        z = sorted([float(v) for v in np.atleast_1d(self.z)], reverse=True)
        da_sliced = da.sel(zt=z, method="nearest")
        da_sliced.attrs.update(da_[0].attrs)

        kws = {}
        if len(base_names) > 1:
            kws["col"] = "base_name"
        if len(z) > 1:
            kws["row"] = "zt"
        if self.field_name.startswith("d_"):
            kws["center"] = 0.0

        g = da_sliced.plot(rasterized=True, robust=True, **kws)

        if self.no_title:
            [ax.set_title("") for ax in np.atleast_1d(g.axes).flatten()]

        plt.tight_layout()
        return g

    def output(self):
        base_names = self.input().keys()
        fn = "{}.{}.png".format("__".join(base_names), self.field_name)
        return luigi.LocalTarget(fn)
