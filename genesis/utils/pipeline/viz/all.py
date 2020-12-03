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
from .... import length_scales
from .... import objects
from ... import plot_types, figure_metadata
from ....utils.calc_flux import scale_flux_to_watts

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


class CumulantScalesProfile(luigi.Task):
    base_names = luigi.Parameter()
    cumulants = luigi.Parameter()
    z_max = luigi.FloatParameter(default=700.0)
    plot_type = luigi.Parameter(default="scales")
    filetype = luigi.Parameter(default="pdf")

    mask = luigi.Parameter(default=None)
    mask_args = luigi.Parameter(default="")

    def _parse_cumulant_arg(self):
        cums = [c.split(":") for c in self.cumulants.split(",")]
        return [c for (n, c) in enumerate(cums) if cums.index(c) == n]

    def requires(self):
        return data.ExtractCumulantScaleProfiles(
            base_names=self.base_names,
            cumulants=self.cumulants,
            mask=self.mask,
            mask_args=self.mask_args,
        )

    def run(self):
        ds = self.input().open()

        cumulants = self._parse_cumulant_arg()
        cumulants_s = ["C({},{})".format(c[0], c[1]) for c in cumulants]

        plot_fn = length_scales.cumulant.vertical_profile.plot.plot

        import ipdb

        with ipdb.launch_ipdb_on_exception():
            plot_fn(data=ds, cumulants=cumulants_s, plot_type=self.plot_type)

        plt.savefig(self.output().path, bbox_inches="tight")

    def output(self):
        base_name = "__".join(self.base_names.split(","))
        fn = length_scales.cumulant.vertical_profile.plot.FN_FORMAT.format(
            base_name=base_name,
            plot_type=self.plot_type,
            mask=self.mask or "no_mask",
            filetype=self.filetype,
        )
        return luigi.LocalTarget(fn)


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


class CumulantSlices(luigi.Task):
    v1 = luigi.Parameter()
    v2 = luigi.Parameter()
    base_names = luigi.Parameter()

    z_step = luigi.IntParameter(default=4)
    z_max = luigi.FloatParameter(default=700.0)
    filetype = luigi.Parameter(default="pdf")

    def requires(self):
        base_names = self.base_names.split(",")

        return dict(
            (
                base_name,
                {
                    self.v1: data.ExtractField3D(
                        field_name=self.v1, base_name=base_name
                    ),
                    self.v2: data.ExtractField3D(
                        field_name=self.v2, base_name=base_name
                    ),
                },
            )
            for base_name in base_names
        )

    def get_suptitle(self, base_name):
        return base_name

    def makeplot(self):
        base_names = self.base_names.split(",")

        datasets = []

        for base_name in base_names:
            inputs = self.input()[base_name]
            v1 = inputs[self.v1].open(decode_times=False)
            v2 = inputs[self.v2].open(decode_times=False)

            ds = xr.merge([v1, v2])
            ds = ds.sel(zt=slice(0.0, self.z_max))
            ds = ds.isel(zt=slice(None, None, self.z_step))
            ds.attrs["name"] = self.get_suptitle(base_name)
            datasets.append(ds)

        plot_fn = length_scales.cumulant.sections.plot
        import ipdb

        with ipdb.launch_ipdb_on_exception():
            ax = plot_fn(
                datasets=datasets,
                var_names=[self.v1, self.v2],
            )

        return ax

    def run(self):
        self.makeplot()
        plt.savefig(self.output().fn, bbox_inches="tight")

    def output(self):
        fn = length_scales.cumulant.sections.FN_FORMAT_PLOT.format(
            v1=self.v1, v2=self.v2, filetype=self.filetype
        )

        return luigi.LocalTarget(fn)


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
        plt.savefig(self.output().fn, bbox_inches="tight")

    def output(self):
        base_names = self.input().keys()
        fn = "{}.{}.png".format("__".join(base_names), self.field_name)
        return luigi.LocalTarget(fn)


class ObjectScalesComparison(luigi.Task):
    plot_definition = luigi.Parameter()
    not_pairgrid = luigi.BoolParameter(default=False)
    file_type = luigi.Parameter(default="png")

    def _parse_plot_definition(self):
        loader = getattr(yaml, "FullLoader", yaml.Loader)
        try:
            with open("{}.yaml".format(self.plot_definition)) as fh:
                return yaml.load(fh, Loader=loader)
        except IOError:
            return yaml.load(self.plot_definition, Loader=loader)

    def requires(self):
        plot_definition = self._parse_plot_definition()

        def _make_dataset_label(**kws):
            arr = []
            for (k, v) in kws.items():
                if k == "object_filters":
                    s = "({})".format(objects.filter.latex_format(v))
                else:
                    s = "{}={}".format(str(k), str(v))
                arr.append(s)
            return ", ".join(arr)

        global_kws = plot_definition["global"]

        variables = set(global_kws.pop("variables").split(","))

        def _merge_kws(g_kws, kws):
            if "object_filters" in g_kws or "object_filters" in kws:
                filter_strings = []
                if "object_filters" in g_kws:
                    filter_strings.append(g_kws["object_filters"])
                    g_kws = dict(g_kws)
                    del g_kws["object_filters"]
                if "object_filters" in kws:
                    filter_strings.append(kws["object_filters"])
                kws["object_filters"] = ",".join(filter_strings)

            kws.update(g_kws)
            return kws

        reqs = {}

        for kws in plot_definition["sources"]:
            if "label" in kws:
                label = kws["label"]
            else:
                label = _make_dataset_label(**kws)
            all_kws = _merge_kws(global_kws, kws)
            target = data.ComputeObjectScales(variables=",".join(variables), **all_kws)
            reqs[label] = target

        return reqs

    def _parse_filters(self, filter_defs):
        if filter_defs is None:
            return []

        ops = {
            ">": lambda f, v: f > v,
            "<": lambda f, v: f < v,
        }
        filters = []

        for filter_s in filter_defs.split(","):
            found_op = False
            for op_str, func in ops.items():
                if op_str in filter_s:
                    field, value = filter_s.split(op_str)
                    filters.append((field, func, value))
                    found_op = True

            if not found_op:
                raise NotImplementedError(filter_s)

        return filters

    def _apply_filters(self, ds, filter_defs):
        for field, func, value in self._parse_filters(filter_defs):
            ds = ds.where(func(ds[field], float(value)))

        return ds

    def _load_data(self):
        def _add_dataset_label(label, input):
            ds = input.open(decode_times=False)
            ds["dataset"] = label
            return ds

        ds = xr.concat(
            [_add_dataset_label(k, input) for (k, input) in self.input().items()],
            dim="dataset",
        )
        plot_definition = self._parse_plot_definition()
        global_params = plot_definition["global"]

        ds = self._apply_filters(ds=ds, filter_defs=global_params.get("filters", None))

        if ds.object_id.count() == 0:
            raise Exception("After filter operations there is nothing to plot!")

        return ds

    def get_suptitle(self):
        plot_definition = self._parse_plot_definition()
        global_params = plot_definition["global"]

        if "object_filters" in global_params:
            object_filters = global_params["object_filters"]
            return objects.filter.latex_format(object_filters)
        else:
            global_params.pop("variables")
            identifier = "\n".join(
                ["{}={}".format(str(k), str(v)) for (k, v) in global_params.items()]
            )
            return identifier

    def run(self):
        ds = self._load_data()

        plot_definition = self._parse_plot_definition()
        global_params = plot_definition["global"]
        variables = global_params.pop("variables").split(",")
        objects.topology.plots.overview(
            ds=ds, as_pairgrid=not self.not_pairgrid, variables=variables
        )

        st = plt.suptitle(self.get_suptitle(), y=[1.1, 1.5][self.not_pairgrid])

        plt.savefig(self.output().fn, bbox_inches="tight", bbox_extra_artists=(st,))

    def output(self):
        fn = "{}.object_scales.{}".format(self.plot_definition, self.file_type)
        return luigi.LocalTarget(fn)


class FilamentarityPlanarityComparison(ObjectScalesComparison):
    reference_shape = luigi.Parameter(default="spheroid")

    def run(self):
        ds = self._load_data()

        def get_new_dataset_label(da):
            base_name = da.dataset.item()
            return xr.DataArray(self.get_base_name_labels().get(base_name, base_name))

        ds.coords["dataset"] = ds.groupby("dataset").apply(get_new_dataset_label)

        import ipdb

        with ipdb.launch_ipdb_on_exception():
            g = objects.topology.plots.filamentarity_planarity(
                ds=ds, reference_shape=self.reference_shape
            )

            g.ax_joint.set_ylim(0.0, 0.95)

        extra_artists = g.fig.get_default_bbox_extra_artists()
        st_str = self.get_suptitle()
        if st_str is not None:
            st = plt.suptitle(st_str, y=[1.05, 1.5][self.not_pairgrid])
            extra_artists.append(st)

        # NOTE: using bbox_extra_artists is disabled because that removes the
        # legend on the joint plot...
        plt.savefig(
            self.output().fn,
            bbox_inches="tight",
        )

    def get_base_name_labels(self):
        return {}

    def output(self):
        fn_base = super().output().fn

        return luigi.LocalTarget(
            fn_base.replace(".object_scales.", ".filamentarity_planarity.")
        )


class ObjectScalesFit(luigi.Task):
    var_name = luigi.Parameter(default="length")
    dv = luigi.FloatParameter(default=None)
    v_max = luigi.FloatParameter(default=None)
    file_type = luigi.Parameter(default="png")
    plot_components = luigi.Parameter(default="default")
    plot_size = luigi.Parameter(default="3,3")

    base_names = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    object_splitting_scalar = luigi.Parameter()
    object_filters = luigi.Parameter(default=None)

    def requires(self):
        return dict(
            [
                (
                    base_name,
                    data.ComputeObjectScales(
                        variables=self.var_name,
                        base_name=base_name,
                        mask_method=self.mask_method,
                        mask_method_extra_args=self.mask_method_extra_args,
                        object_splitting_scalar=self.object_splitting_scalar,
                        object_filters=self.object_filters,
                    ),
                )
                for base_name in self.base_names.split(",")
            ]
        )

    def get_base_name_labels(self):
        return {}

    def get_suptitle(self):
        s_filters = objects.filter.latex_format(self.object_filters)
        return "{}\n{}".format(self.base_names, s_filters)

    def run(self):
        inputs = self.input()
        if self.plot_components == "default":
            plot_components = "default"
            Nc = 4
        else:
            plot_components = self.plot_components.split(",")
            Nc = len(plot_components)

        sx, sy = [float(v) for v in self.plot_size.split(",")]
        fig, axes = plt.subplots(
            ncols=Nc,
            nrows=len(inputs),
            figsize=(sx * Nc, sy * len(inputs)),
            sharex="col",
            sharey="col",
        )

        if len(axes.shape) == 1:
            axes = np.array([axes])

        for n, (base_name, input) in enumerate(inputs.items()):
            input = input.open()
            if isinstance(input, xr.Dataset):
                ds = input
                da_v = ds[self.var_name]
            else:
                da_v = input
            if self.dv is not None:
                da_v = da_v[da_v > self.dv]
            else:
                da_v = da_v[da_v > 0.0]

            plot_to = axes[n]
            length_scales.model_fitting.exponential_fit.fit(
                da_v,
                dv=self.dv,
                debug=False,
                plot_to=plot_to,
                plot_components=plot_components,
            )

            ax = plot_to[0]
            desc = self.get_base_name_labels().get(base_name)
            if desc is None:
                desc = base_name.replace("_", " ").replace(".", " ")
            desc += "\n({} objects)".format(len(da_v.object_id))
            ax.text(
                -0.5, 0.5, desc, transform=ax.transAxes, horizontalalignment="right"
            )

        sns.despine()

        if self.v_max:
            [ax.set_xlim(0, self.v_max) for ax in axes[:, :2].flatten()]
        if da_v.units == "m":
            [ax.set_ylim(1.0e-6, None) for ax in axes[:, 1].flatten()]
        if axes.shape[0] > 1:
            [ax.set_xlabel("") for ax in axes[0].flatten()]
        [ax.set_title("") for ax in axes[1:].flatten()]
        plt.suptitle(self.get_suptitle(), y=1.1)
        plt.tight_layout()
        plt.savefig(self.output().fn, bbox_inches="tight")

    def output(self):
        s_filter = ""
        if self.object_filters is not None:
            s_filter = ".filtered_by.{}".format(
                (
                    self.object_filters.replace(",", ".")
                    .replace(":", "__")
                    .replace("=", "_")
                )
            )
        fn = "object_scales_exp_fit.{}.{}{}.{}".format(
            self.var_name, self.base_names.replace(",", "__"), s_filter, self.file_type
        )
        return luigi.LocalTarget(fn)


class ObjectsScaleDist(luigi.Task):
    var_name = luigi.Parameter()
    dv = luigi.FloatParameter(default=None)
    v_max = luigi.FloatParameter(default=None)
    file_type = luigi.Parameter(default="png")
    show_cumsum = luigi.BoolParameter(default=False)
    cumsum_markers = luigi.Parameter(default=None)
    as_density = luigi.Parameter(default=False)
    figsize = luigi.Parameter(default="6,6")

    base_names = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    object_splitting_scalar = luigi.Parameter()
    object_filters = luigi.Parameter(default=None)

    def requires(self):
        reqs = {}
        for var_name in self.var_name.split(","):
            reqs[var_name] = dict(
                [
                    (
                        base_name,
                        data.ComputeObjectScales(
                            variables=self.var_name,
                            base_name=base_name,
                            mask_method=self.mask_method,
                            mask_method_extra_args=self.mask_method_extra_args,
                            object_splitting_scalar=self.object_splitting_scalar,
                            object_filters=self.object_filters,
                        ),
                    )
                    for base_name in self.base_names.split(",")
                ]
            )
        return reqs

    @staticmethod
    def _calc_fixed_bin_args(v, dv):
        vmin = np.floor(v.min() / dv) * dv
        vmax = np.ceil(v.max() / dv) * dv
        nbins = int((vmax - vmin) / dv)
        return dict(range=(vmin, vmax), bins=nbins)

    def get_base_name_labels(self):
        return {}

    def get_title(self):
        return ""

    def run(self):  # noqa
        figsize = [float(v) for v in self.figsize.split(",")]
        N_vars = len(self.var_name.split(","))
        fig, axes = plt.subplots(
            figsize=(figsize[0] * N_vars, figsize[1]), ncols=N_vars, sharey=True
        )
        if N_vars == 1:
            axes = [
                axes,
            ]

        d_units = []
        for n, (var_name, inputs) in enumerate(self.input().items()):
            ax = axes[n]

            if self.show_cumsum:
                ax_twin = ax.twinx()

            bins = None
            for n, (base_name, input) in enumerate(inputs.items()):
                input = input.open()
                if isinstance(input, xr.Dataset):
                    ds = input
                    da_v = ds[var_name]
                else:
                    da_v = input

                da_v = da_v[np.logical_and(~np.isnan(da_v), ~np.isinf(da_v))]
                desc = self.get_base_name_labels().get(base_name)
                if desc is None:
                    desc = base_name.replace("_", " ").replace(".", " ")
                desc += " ({} objects)".format(int(da_v.object_id.count()))
                kws = dict(density=self.as_density)
                if self.dv is not None:
                    kws.update(self._calc_fixed_bin_args(v=da_v.values, dv=self.dv))
                if bins is not None:
                    kws["bins"] = bins
                _, bins, pl_hist = da_v.plot.hist(ax=ax, alpha=0.4, label=desc, **kws)

                if self.show_cumsum:
                    # cumulative dist. plot
                    x_ = np.sort(da_v)
                    y_ = np.cumsum(x_)
                    c = pl_hist[0].get_facecolor()
                    ax_twin.plot(
                        x_,
                        y_,
                        color=c,
                        marker=".",
                        linestyle="",
                        markeredgecolor="None",
                    )
                    ax_twin.axhline(y=y_[-1], color=c, linestyle="--", alpha=0.3)

                    if self.cumsum_markers is not None:
                        markers = [float(v) for v in self.cumsum_markers.split(",")]
                        for m in markers:
                            i = np.nanargmin(np.abs(m * y_[-1] - y_))
                            x_m = x_[i]
                            ax_twin.axvline(x_m, color=c, linestyle=":")

            ax.set_title(self.get_title())
            if self.as_density:
                ax.set_ylabel("object density [1/{}]".format(da_v.units))
            else:
                ax.set_ylabel("num objects")
            if self.show_cumsum:
                ax_twin.set_ylabel(
                    "sum of {}".format(xr.plot.utils.label_from_attrs(da_v))
                )

            d_units.append(da_v.units)

        if all([d_units[0] == u for u in d_units[1:]]):
            ax1 = axes[0]
            [ax1.get_shared_x_axes().join(ax1, ax) for ax in axes[1:]]
            ax1.autoscale()
            [ax.set_ylabel("") for ax in axes[1:]]

        bbox_extra_artists = []

        sns.despine(fig)

        st_str = self.get_suptitle()
        if st_str is not None:
            st = plt.suptitle(st_str, y=1.1)
            bbox_extra_artists.append(st)

        ax_lgd = axes[len(axes) // 2]
        lgd = plt.figlegend(
            *ax_lgd.get_legend_handles_labels(),
            loc="lower center",
        )

        plot_types.adjust_fig_to_fit_figlegend(
            fig=fig, figlegend=lgd, direction="bottom"
        )
        bbox_extra_artists.append(lgd)

        if self.v_max is not None:
            ax.set_xlim(0.0, self.v_max)

        plt.savefig(
            self.output().fn, bbox_inches="tight", bbox_extra_artists=bbox_extra_artists
        )

    def get_suptitle(self):
        s_filters = objects.filter.latex_format(self.object_filters)
        return plt.suptitle("{}\n{}".format(self.base_names, s_filters), y=1.1)

    def output(self):
        s_filter = ""
        if self.object_filters is not None:
            s_filter = ".filtered_by.{}".format(
                (
                    self.object_filters.replace(",", ".")
                    .replace(":", "__")
                    .replace("=", "_")
                )
            )
        fn = "objects_scale_dist.{}.{}{}.{}".format(
            self.var_name.replace(",", "__"),
            self.base_names.replace(",", "__"),
            s_filter,
            self.file_type,
        )
        return luigi.LocalTarget(fn)


class ObjectsScalesJointDist(luigi.Task):
    x = luigi.Parameter()
    y = luigi.Parameter()
    file_type = luigi.Parameter(default="png")

    xmax = luigi.FloatParameter(default=None)
    ymax = luigi.FloatParameter(default=None)
    plot_type = luigi.Parameter(default="scatter")
    plot_aspect = luigi.FloatParameter(default=None)
    plot_annotations = luigi.Parameter(default=None)
    scaling = luigi.Parameter(default=None)

    base_names = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    object_splitting_scalar = luigi.Parameter()
    object_filters = luigi.Parameter(default=None)

    def requires(self):
        return dict(
            [
                (
                    base_name,
                    data.ComputeObjectScales(
                        variables="{},{}".format(self.x, self.y),
                        base_name=base_name,
                        mask_method=self.mask_method,
                        mask_method_extra_args=self.mask_method_extra_args,
                        object_splitting_scalar=self.object_splitting_scalar,
                        object_filters=self.object_filters,
                    ),
                )
                for base_name in self.base_names.split(",")
            ]
        )

    @staticmethod
    def _calc_fixed_bin_args(v, dv):
        vmin = np.floor(v.min() / dv) * dv
        vmax = np.ceil(v.max() / dv) * dv
        nbins = int((vmax - vmin) / dv)
        return dict(range=(vmin, vmax), bins=nbins)

    def get_suptitle(self):
        s_filters = objects.filter.latex_format(self.object_filters)
        return "{}\n{}".format(self.base_names, s_filters)

    def get_base_name_labels(self):
        return {}

    def make_plot(self):  # noqa
        inputs = self.input()

        kws = {}
        if self.xmax is not None:
            kws["xlim"] = (0, self.xmax)
        if self.ymax is not None:
            kws["ylim"] = (0, self.ymax)

        if self.plot_type.startswith("jointplot"):
            if "_" in self.plot_type:
                kws["joint_type"] = self.plot_type.split("_")[-1]

            def _lab(base_name, ds_):
                ds_["dataset"] = self.get_base_name_labels().get(base_name, base_name)
                return ds_

            dss = [
                _lab(base_name, input.open()) for (base_name, input) in inputs.items()
            ]
            ds = xr.concat(dss, dim="dataset")

            g = plot_types.multi_jointplot(
                x=self.x, y=self.y, z="dataset", ds=ds, lgd_ncols=2, **kws
            )
            s_title = self.get_suptitle()
            if s_title is not None:
                plt.suptitle(s_title, y=1.1)
            ax = g.ax_joint
            if self.plot_aspect is not None:
                raise Exception(
                    "Can't set aspect ratio on jointplot, set limits instead"
                )
        elif self.plot_type in ["scatter", "scatter_hist"]:
            ax = None
            alpha = 1.0 / len(inputs)
            if ax is None:
                fig, ax = plt.subplots()
            for n, (base_name, input) in enumerate(inputs.items()):
                print(base_name)
                ds = input.open()
                da_v1 = ds[self.x]
                da_v2 = ds[self.y]

                desc = base_name.replace("_", " ").replace(".", " ")
                desc += " ({} objects)".format(len(da_v1))
                if self.plot_type == "scatter":
                    ax.scatter(
                        x=da_v1.values, y=da_v2.values, alpha=alpha, label=desc, s=5.0
                    )
                else:
                    ax = plot_types.make_marker_plot(
                        x=da_v1.values, y=da_v2.values, alpha=alpha
                    )

            ax.set_xlabel(xr.plot.utils.label_from_attrs(da_v1))
            ax.set_ylabel(xr.plot.utils.label_from_attrs(da_v2))
            sns.despine()
            ax.legend()
            if self.plot_aspect is not None:
                ax.set_aspect(self.plot_aspect)

            plt.title("{}\n{}".format(self.base_names, self.object_filters))
            if self.xmax is not None:
                ax.set_xlim(np.nanmin(da_v1), self.xmax)
            else:
                xmax = np.nanmax(da_v1)
                xmin = np.nanmin(da_v1)
                ax.set_xlim(xmin, xmax)
            if self.ymax is not None:
                ymin = [0.0, None][np.nanmin(da_v2) < 0.0]
                ax.set_ylim(ymin, self.ymax)
        else:
            raise NotImplementedError(self.plot_type)

        # for n, (base_name, input) in enumerate(inputs.items()):
        # ds = input.open()
        # da_v1 = ds[self.x]
        # da_v2 = ds[self.y]

        # desc = base_name.replace('_', ' ').replace('.', ' ')
        # if hue_label:
        # ds_ = ds.where(ds[hue_label], drop=True)
        # ds_[v].plot.hist(ax=ax, bins=bins)
        # g = sns.jointplot(x=self.x, y=self.y, , s=10)

        # ax = g.ax_joint
        # ax.set_xlabel(xr.plot.utils.label_from_attrs(da_v1))
        # ax.set_ylabel(xr.plot.utils.label_from_attrs(da_v2))

        if self.plot_annotations is not None:
            for annotation in self.plot_annotations.split(","):
                if (
                    annotation == "plume_vs_thermal"
                    and self.x == "z_proj_length"
                    and self.y == "z_min"
                ):
                    z_b = 200.0
                    z_cb = 600.0
                    if self.xmax is None:
                        xmax = ax.get_xlim()[1]
                    else:
                        xmax = self.xmax
                    x_ = np.linspace(z_cb - z_b, xmax, 100)

                    ax.plot(
                        x_,
                        z_cb - x_,
                        marker="",
                        color="grey",
                        linestyle="--",
                        alpha=0.6,
                    )
                    t_kws = dict(
                        transform=ax.transData,
                        color="grey",
                        horizontalalignment="center",
                    )
                    ax.text(356.0, 100.0, "thermals", **t_kws)
                    ax.text(650.0, 100.0, "plumes", **t_kws)
                elif annotation == "unit_line":
                    x_ = np.linspace(
                        max(ax.get_xlim()[0], ax.get_ylim()[0]),
                        min(ax.get_xlim()[-1], ax.get_ylim()[-1]),
                        100,
                    )
                    ax.plot(x_, x_, linestyle="--", alpha=0.6, color="grey")
                else:
                    raise NotImplementedError(annotation, self.x, self.y)

        if self.scaling is None:
            pass
        elif self.scaling == "loglog":
            ax.set_yscale("log")
            ax.set_xscale("log")
        else:
            raise NotImplementedError(self.scaling)

        plt.tight_layout()

        return ax

    def run(self):
        try:
            plt.savefig(self.output().fn, bbox_inches="tight")
        except IOError:
            import hues
            import uuid

            fn = "plot.{}.{}".format(str(uuid.uuid4()), self.file_type)
            hues.warn("filename became to long, saved to `{}`".format(fn))
            plt.savefig(fn, bbox_inches="tight")

    def output(self):
        s_filter = ""
        if self.object_filters is not None:
            s_filter = ".filtered_by.{}".format(
                (
                    self.object_filters.replace(",", ".")
                    .replace(":", "__")
                    .replace("=", "_")
                )
            )

        objects_name = data.IdentifyObjects.make_name(
            base_name=self.base_names,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            filter_defs=self.object_filters,
        )

        fn = "objects_scales_joint_dist.{}__{}.{}{}.{}".format(
            self.x, self.y, objects_name.replace(",", "__"), s_filter, self.file_type
        )
        return luigi.LocalTarget(fn)


class ObjectScaleVsHeightComposition(luigi.Task):
    x = luigi.Parameter()
    field_name = luigi.Parameter()

    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    object_splitting_scalar = luigi.Parameter()

    object_filters = luigi.Parameter(default=None)
    dx = luigi.FloatParameter(default=None)
    z_max = luigi.FloatParameter(default=None)
    filetype = luigi.Parameter(default="png")
    x_max = luigi.FloatParameter(default=None)

    scale_by = luigi.OptionalParameter(default=None)
    object_filters = luigi.Parameter(default=None)

    # make it possible to add an extra profile for filtered objects
    ref_profile_object_filters = luigi.Parameter(default=None)

    add_profile_legend = luigi.BoolParameter(default=True)
    include_mask_profile = luigi.BoolParameter(default=True)
    fig_width = luigi.FloatParameter(default=7.0)

    def requires(self):
        kwargs = dict(
            base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            field_name=self.field_name,
            z_max=self.z_max,
            x=self.x,
        )

        reqs = dict(
            base=data.ComputeObjectScaleVsHeightComposition(
                object_filters=self.object_filters,
                **kwargs,
            )
        )

        if self.ref_profile_object_filters is not None:
            reqs["extra_ref"] = data.ComputeObjectScaleVsHeightComposition(
                object_filters=self.ref_profile_object_filters, **kwargs
            )

        return reqs

    def run(self):
        if self.ref_profile_object_filters is None:
            ds = self.input()["base"].open()
            mean_profile_components = ["full domain", "objects"]
        else:
            ds_base = self.input()["base"].open()

            sampling_labels = [
                d if not d == "objects" else "all objects"
                for d in ds_base.sampling.values
            ]
            ds_mean_base = ds_base.drop_dims(["object_id"]).assign_coords(
                sampling=sampling_labels
            )

            ds_objects = ds_base.drop_dims(["sampling"])

            s_filters = objects.filter.latex_format(self.ref_profile_object_filters)
            ds_mean_filtered = (
                self.input()["extra_ref"]
                .open()
                .sel(sampling="objects")
                .assign_coords(sampling=[f"{s_filters} objects"])
                .drop_dims(["object_id"])
            )

            ds_mean = xr.concat([ds_mean_base, ds_mean_filtered], dim="sampling")
            mean_profile_components = ds_mean.sampling.values.tolist()
            ds = xr.merge([ds_mean, ds_objects])
            ds.attrs.update(ds_base.attrs)

        if not self.include_mask_profile:
            mean_profile_components.remove("mask")

        if self.scale_by is not None:
            scaling_factors = {}
            if ":" in self.scale_by:
                raise NotImplementedError
            else:
                for dim in ds.dims:
                    scaling_factors[dim] = float(self.scale_by)

        import ipdb

        with ipdb.launch_ipdb_on_exception():
            ax = objects.flux_contribution.plot(
                ds=ds,
                x=self.x,
                v=self.field_name,
                dx=self.dx,
                mean_profile_components=mean_profile_components,
                add_profile_legend=self.add_profile_legend,
                fig_width=self.fig_width,
                # scaling_factors=scaling_factors
            )

        if self.x_max is not None:
            ax.set_xlim(0.0, self.x_max)

        N_objects = int(ds.object_id.count())
        plt.suptitle(self.get_suptitle(N_objects=N_objects), y=1.0)

        plt.savefig(self.output().fn, bbox_inches="tight")

    def get_suptitle(self, N_objects):
        s_filters = objects.filter.latex_format(self.object_filters)
        return "{} ({} objects)\n{}".format(self.base_name, N_objects, s_filters)

    def output(self):
        mask_name = data.MakeMask.make_mask_name(
            base_name=self.base_name,
            method_name=self.mask_method,
            method_extra_args=self.mask_method_extra_args,
        )
        s_filter = ""
        if self.object_filters is not None:
            s_filter = ".filtered_by.{}".format(
                (
                    self.object_filters.replace(",", ".")
                    .replace(":", "__")
                    .replace("=", "_")
                )
            )
        fn = (
            "{base_name}.{mask_name}.{field_name}__by__{x}"
            "{s_filter}.{filetype}".format(
                base_name=self.base_name,
                mask_name=mask_name,
                field_name=self.field_name,
                x=self.x,
                filetype=self.filetype,
                s_filter=s_filter,
            )
        )
        target = luigi.LocalTarget(fn)
        return target


def snake_case_class_name(obj):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", obj.__class__.__name__).lower()


class FluxFractionCarried(luigi.Task):
    base_name = luigi.Parameter()
    scalar = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    object_splitting_scalar = luigi.Parameter()
    object_filters = luigi.Parameter(default=None)
    z_max = luigi.Parameter(default=600)

    def requires(self):
        return data.ComputeObjectScaleVsHeightComposition(
            base_name=self.base_name,
            field_name="{}_flux".format(self.scalar),
            z_max=self.z_max,
            x="r_equiv",
            object_filters=self.object_filters,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
        )

    def get_suptitle(self):
        if self.object_filters is not None:
            return objects.filter.latex_format(self.object_filters)
        else:
            return "all objects"

    def run(self):
        ds = input.open()
        fig, axes = self._make_plot(ds, self.output().fn)
        fig.savefig(self.output().fn, bbox_inches="tight")

    def _make_plot(self, ds, **kwargs):
        ds_ = ds.rename({"{}_flux__mean".format(self.scalar): "flux_mean", "zt": "z"})
        ds_ = ds_[["areafrac", "flux_mean"]]
        ds_.attrs["scalar"] = self.scalar
        ds_["flux_mean"] = scale_flux_to_watts(da=ds_["flux_mean"], scalar=self.scalar)
        fig, axes = objects.flux_contribution.plot_with_areafrac(ds=ds_, **kwargs)
        suptitle = self.get_suptitle()
        if suptitle:
            plt.suptitle(self.get_suptitle(), y=1.1)
        return fig, axes

    def output(self):
        s_filter = ""
        if self.object_filters is not None:
            s_filter = "filtered_by.{}".format(
                (
                    self.object_filters.replace(",", ".")
                    .replace(":", "__")
                    .replace("=", "_")
                )
            )
        else:
            s_filter = "all_objects"

        fn_plot = "{}.{}.{}.png".format(
            snake_case_class_name(self), self.base_name, s_filter
        )
        return luigi.LocalTarget(fn_plot)


class FluxFractionCarriedFiltersComparison(FluxFractionCarried):
    def requires(self):
        assert ";" in self.object_filters
        object_filters_sets = self.object_filters.split(";")
        return {
            object_filters: data.ComputeObjectScaleVsHeightComposition(
                base_name=self.base_name,
                field_name="{}_flux".format(self.scalar),
                z_max=self.z_max,
                x="r_equiv",
                object_filters=object_filters,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                object_splitting_scalar=self.object_splitting_scalar,
            )
            for object_filters in object_filters_sets
        }

    def run(self):
        input = self.input()
        dss = []
        for n, (object_filters, input) in enumerate(self.input().items()):
            ds_ = input.open()
            flux_var = "{}_flux__mean".format(self.scalar)
            vars_needed = ["areafrac", flux_var]
            ds_ = ds_[vars_needed]

            if n > 0:
                sampling_modes = ds_.sampling.values.tolist()
                sampling_modes.remove("full domain")
                sampling_modes.remove("mask")
                ds_ = ds_.sel(sampling=sampling_modes)

            obj_s_latex = objects.filter.latex_format(object_filters)
            ds_["sampling"] = ds_.sampling.where(
                ds_.sampling != "objects", ds_.sampling.values + ": " + obj_s_latex
            )
            dss.append(ds_)

        ds = xr.concat(dss, dim="sampling")
        ds.attrs["scalar"] = self.scalar

        fig, axes = self._make_plot(ds, figsize=(5, 3), legend_ncols=1)
        fig.savefig(self.output().fn, bbox_inches="tight")

    def get_suptitle(self):
        return None


class Suite(luigi.Task):
    base_name = luigi.Parameter(default=None)
    timestep = luigi.IntParameter(default=None)

    DISTS = dict(
        mean_profile="mean.profiles",
        cross_section="cross.sections",
        cumulant_profiles="cumulant.profiles",
    )

    CROSS_SECTION_VARS = ["qv", "w", "qc", "cvrxp_p_stddivs"]

    def requires(self):
        reqs = {}
        if self.timestep is not None:

            def add_timestep(bn):
                return "{}.tn{}".format(bn, self.timestep)

        else:

            def add_timestep(bn):
                return bn

        if self.base_name is None:
            datasources = data.get_datasources()
            for base_name in datasources.keys():
                reqs["subsuite__{}".format(base_name)] = Suite(
                    base_name=add_timestep(base_name)
                )

            for v in Suite.CROSS_SECTION_VARS:
                base_names = ",".join([add_timestep(bn) for bn in datasources.keys()])
                reqs["cross_section__{}".format(v)] = CrossSection(
                    base_names=base_names, var_name=v, z="100.,400.,600.,800."
                )
                reqs["cumulant_profiles__{}".format(v)] = CumulantScalesProfile(
                    base_names=base_names,
                    cumulants="w:w,qv:qv,qc:qc,theta_l:theta_l,cvrxp:cvrxp,w:qv,w:qc,w:cvrxp",
                    z_max=1000.0,
                )
        else:
            reqs["mean_profile"] = HorizontalMeanProfile(
                base_name=self.base_name,
            )
        return reqs

    def run(self):
        for comp, target in self.input().items():
            if comp.startswith("subsuite__"):
                continue
            dst_path = self._build_output_path(comp=comp, target=target)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(target.fn, dst_path)

    def _build_output_path(self, comp, target):
        d = self.DISTS[comp.split("__")[0]]
        return Path(d) / target.fn

    def output(self):
        outputs = []
        for (comp, target) in self.input().items():
            if comp.startswith("subsuite__"):
                outputs.append(target)
            else:
                p = self._build_output_path(comp=comp, target=target)
                moved_target = luigi.LocalTarget(p)
                outputs.append(moved_target)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
