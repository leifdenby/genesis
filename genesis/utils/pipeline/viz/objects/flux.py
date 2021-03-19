import re

import luigi
import xarray as xr
import matplotlib.pyplot as plt

from ... import data
from ..... import objects
from .....utils.calc_flux import scale_flux_to_watts


class ObjectScaleVsHeightComposition(luigi.Task):
    """
    Decompose `field_name` by height and object property `x`, with number and
    mean-flux distributions shown as margin plots. An extract reference profile
    can by defining the filters on the objects to consider with
    `ref_profile_object_filters`
    """
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

    def make_plot(self):
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

        g = objects.flux_contribution.plot(
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
            g.ax_joint.set_xlim(0.0, self.x_max)

        N_objects = int(ds.object_id.count())
        plt.suptitle(self.get_suptitle(N_objects=N_objects), y=1.0)

        return g

    def run(self):
        self.make_plot()
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
        fig, _ = self._make_plot(ds, self.output().fn)
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


def snake_case_class_name(obj):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", obj.__class__.__name__).lower()


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
