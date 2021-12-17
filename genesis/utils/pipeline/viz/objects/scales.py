"""
Tasks for producing plots of object scales
"""
import luigi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
import yaml

from ..... import length_scales, objects
from .... import plot_types
from ... import data


class ObjectScalesComparison(luigi.Task):
    plot_definition = luigi.Parameter()
    not_pairgrid = luigi.BoolParameter(default=False)
    file_type = luigi.Parameter(default="png")

    def _parse_plot_definition(self):
        if type(self.plot_definition) == str and self.plot_definition.endswith(".yaml"):
            loader = getattr(yaml, "FullLoader", yaml.Loader)
            try:
                with open("{}.yaml".format(self.plot_definition)) as fh:
                    return yaml.load(fh, Loader=loader)
            except IOError:
                pass

        if type(self.plot_definition) == dict:
            return self.plot_definition

        loader = getattr(yaml, "FullLoader", yaml.Loader)
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
            import uuid

            import hues

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
