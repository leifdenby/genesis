import hashlib
from functools import partial
from pathlib import Path

import dask_image
import luigi
import numpy as np
import xarray as xr

from .... import length_scales, objects
from ....objects import integral_properties
from ... import make_mask
from .base import XArrayTarget, get_workdir
from .extraction import ExtractField3D
from .masking import MakeMask
from .tracking_2d import PerformObjectTracking2D


def merge_object_datasets(dss):
    def _strip_coord(ds_):
        """
        remove the values of the `object_id` coordinate so that
        we can concate along it without having duplicates. We keep
        a copy of the original object id values
        """
        obj_id = ds_["object_id"]
        del ds_["object_id"]
        ds_["org_object_id"] = ("object_id"), obj_id.values
        return ds_

    dss = [_strip_coord(ds_) for ds_ in dss]
    ds = xr.concat(dss, dim="object_id")
    ds["object_id"] = np.arange(ds.object_id.max() + 1)

    return ds


class IdentifyObjects(luigi.Task):
    splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    filters = luigi.Parameter(default=None)

    def requires(self):
        if self.filters is not None:
            return FilterObjects(
                object_splitting_scalar=self.splitting_scalar,
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                filter_defs=self.filters,
            )
        else:
            return dict(
                mask=MakeMask(
                    base_name=self.base_name,
                    method_name=self.mask_method,
                    method_extra_args=self.mask_method_extra_args,
                ),
                scalar=ExtractField3D(
                    base_name=self.base_name, field_name=self.splitting_scalar
                ),
            )

    def run(self):
        if self.filters is not None:
            pass
        else:
            da_mask = xr.open_dataarray(self.input()["mask"].fn).squeeze()
            da_scalar = xr.open_dataarray(self.input()["scalar"].fn).squeeze()

            object_labels = objects.identify.label_objects(
                mask=da_mask, splitting_scalar=da_scalar
            )

            object_labels.to_netcdf(self.output().fn)

    @staticmethod
    def make_name(
        base_name,
        mask_method,
        mask_method_extra_args,
        object_splitting_scalar,
        filter_defs,
    ):
        mask_name = MakeMask.make_mask_name(
            base_name=base_name,
            method_name=mask_method,
            method_extra_args=mask_method_extra_args,
        )
        objects_name = objects.identify.make_objects_name(
            mask_name=mask_name, splitting_var=object_splitting_scalar
        )
        if filter_defs is not None:
            s_filters = (
                filter_defs.replace(",", ".").replace("=", "").replace(":", "__")
            )
            objects_name = "{}.filtered_by.{}".format(objects_name, s_filters)
        return objects_name

    def output(self):
        if self.filters is not None:
            return self.input()

        objects_name = self.make_name(
            base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.splitting_scalar,
            filter_defs=self.filters,
        )

        fn = objects.identify.OUT_FILENAME_FORMAT.format(
            base_name=self.base_name, objects_name=objects_name
        )
        p = get_workdir() / self.base_name / fn

        return XArrayTarget(str(p))


class ComputeObjectScales(luigi.Task):
    """
    Compute scales given by `variables` (multiple can be given by
    comma-separation, e.g. variables=`r_equiv,qv_flux__mean`) on objects
    defined by `mask_method` (optionally with extra mask method arguments with
    `mask_method_extra_args`) and splitting-scalar `object_splitting_scalar`.
    """

    object_splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    variables = luigi.Parameter(default="com_angles")
    object_filters = luigi.Parameter(default=None)

    def requires(self):
        if "+" in self.base_name:
            # # "+" can be used a shorthand to concat objects together from
            # # different datasets
            base_names = self.base_name.split("+")
            reqs = dict(
                [
                    (
                        base_name,
                        ComputeObjectScales(
                            base_name=base_name,
                            mask_method=self.mask_method,
                            mask_method_extra_args=self.mask_method_extra_args,
                            variables=self.variables,
                            object_splitting_scalar=self.object_splitting_scalar,
                            object_filters=self.object_filters,
                        ),
                    )
                    for base_name in base_names
                ]
            )
            return reqs
        elif self.object_filters is not None:
            return FilterObjectScales(
                object_splitting_scalar=self.object_splitting_scalar,
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                variables=self.variables,
                object_filters=self.object_filters,
            )
        else:
            return self._requires_single()

    def _requires_single(self):
        variables = set(self.variables.split(","))
        reqs = []

        # some methods provide more than one variable so we map them to the
        # correct method here
        variables_mapped = []
        for v in variables:
            v_mapped = integral_properties.VAR_MAPPINGS.get(v)
            if v_mapped is not None:
                variables_mapped.append(v_mapped)
            else:
                variables_mapped.append(v)
        # we only want to call each method once
        variables_mapped = set(variables_mapped)

        for v in variables_mapped:
            if v == "minkowski":
                reqs.append(
                    ComputeObjectMinkowskiScales(
                        base_name=self.base_name,
                        object_splitting_scalar=self.object_splitting_scalar,
                        mask_method=self.mask_method,
                        mask_method_extra_args=self.mask_method_extra_args,
                        object_filters=self.object_filters,
                    )
                )
            else:
                assert self.object_filters is None
                reqs.append(
                    ComputeObjectScale(
                        base_name=self.base_name,
                        variable=v,
                        object_splitting_scalar=self.object_splitting_scalar,
                        mask_method=self.mask_method,
                        mask_method_extra_args=self.mask_method_extra_args,
                        # object_filters=self.object_filters,
                    )
                )

        return reqs

    def run(self):
        if "+" not in self.base_name and self.object_filters is not None:
            pass
        else:
            if "+" in self.base_name:
                dss = [
                    fh.open(decode_times=False)
                    for (base_name, fh) in self.input().items()
                ]
                ds = merge_object_datasets(dss=dss)

                p_data = Path(self.output().fn).parent
                p_data.mkdir(parents=True, exist_ok=True)
            else:
                ds = xr.merge(
                    [input.open(decode_times=False) for input in self.input()]
                )

            objects_name = IdentifyObjects.make_name(
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                object_splitting_scalar=self.object_splitting_scalar,
                filter_defs=self.object_filters,
            )
            ds.attrs["base_name"] = self.base_name
            ds.attrs["objects_name"] = objects_name

            if isinstance(ds, xr.Dataset):
                ds = ds[self.variables.split(",")]
            else:
                assert ds.name == self.variables

            ds.to_netcdf(self.output().fn)

    def output(self):
        if "+" not in self.base_name and self.object_filters is not None:
            return self.input()

        objects_name = IdentifyObjects.make_name(
            base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            filter_defs=self.object_filters,
        )

        s = self.variables.encode("utf-8")
        scales_identifier = hashlib.md5(s).hexdigest()
        fn = objects.integrate.FN_OUT_FORMAT.format(
            base_name=self.base_name,
            objects_name=objects_name,
            name="object_scales.{}".format(scales_identifier),
        )

        # XXX: filenames are max allowed to be 255 characters on linux, this is
        # a hack to generate a unique filename we can use
        if len(fn) > 255:
            fn = "{}.nc".format(hashlib.md5(fn.encode("utf-8")).hexdigest())

        p = get_workdir() / self.base_name / fn
        target = XArrayTarget(str(p))

        if target.exists():
            ds = target.open(decode_times=False)
            variables = self.variables.split(",")
            if isinstance(ds, xr.Dataset):
                if not set(ds.data_vars) == set(variables):
                    raise Exception(
                        f"There's a problem with existing file {p}: "
                        f"{set(ds.data_vars)} != {set(variables)}"
                    )
            else:
                assert ds.name == variables[0]

        return target


class FilterObjectScales(ComputeObjectScales):
    object_splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    variables = luigi.Parameter(default="num_cells")
    object_filters = luigi.Parameter()

    def requires(self):
        filters = objects.filter.parse_defs(self.object_filters)
        variables = self.variables.split(",")
        for filter in filters:
            variables += filter["reqd_props"]

        reqs = {}
        reqs["objects_scales"] = ComputeObjectScales(
            variables=",".join(variables),
            base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            # no object filter, get properties for all objects
        )

        for filter in filters:
            if "cloud_tracking_data" in filter["extra"]:
                reqs["tracking"] = PerformObjectTracking2D(
                    base_name=self.base_name,
                    tracking_type=objects.filter.TrackingType.THERMALS_ONLY,
                )
            if "objects_3d" in filter["extra"]:
                reqs["objects"] = IdentifyObjects(
                    base_name=self.base_name,
                    mask_method=self.mask_method,
                    mask_method_extra_args=self.mask_method_extra_args,
                    splitting_scalar=self.object_splitting_scalar,
                )

        return reqs

    def run(self):
        input = self.input()
        ds_objects_scales = input["objects_scales"].open()
        if not isinstance(ds_objects_scales, xr.Dataset):
            # if only one variable was requested we'll get a dataarray back
            ds_objects_scales = ds_objects_scales.to_dataset()

        filters = objects.filter.parse_defs(self.object_filters)
        for filter in filters:
            fn = filter["fn"]
            kws = {}
            if "cloud_tracking_data" in filter["extra"]:
                cloud_data = self.requires()["tracking"].get_cloud_data()
                kws["cloud_tracking_data"] = cloud_data
            if "objects_3d" in filter["extra"]:
                da_objects = self.input()["objects"].open(decode_times=False)
                kws["da_objects"] = da_objects

            if "extra_kws" in filter:
                kws.update(**filter["extra_kws"])

            ds_objects_scales = fn(ds_objects_scales, **kws)

        objects_name = IdentifyObjects.make_name(
            base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            filter_defs=self.object_filters,
        )
        ds_objects_scales.attrs["base_name"] = self.base_name
        ds_objects_scales.attrs["objects_name"] = objects_name

        ds_objects_scales.to_netcdf(self.output().fn)

    def output(self):
        objects_name = IdentifyObjects.make_name(
            base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            filter_defs=self.object_filters,
        )

        fn = objects.integrate.FN_OUT_FORMAT.format(
            base_name=self.base_name,
            objects_name=objects_name,
            name="object_scales_collection",
        )

        p = get_workdir() / self.base_name / fn
        target = XArrayTarget(str(p))

        if target.exists():
            ds = target.open(decode_times=False)
            variables = self.variables.split(",")
            if isinstance(ds, xr.Dataset):
                if any([v not in ds.data_vars for v in variables]):
                    p.unlink()
            elif ds.name != self.variables:
                print(ds.name)
                p.unlink()

        return target


class FilterObjects(luigi.Task):
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    object_splitting_scalar = luigi.Parameter()
    filter_defs = luigi.Parameter()

    def requires(self):
        filters = self._parse_filter_defs()
        reqs = dict(props={}, extra={})
        reqs["objects"] = IdentifyObjects(
            base_name=self.base_name,
            splitting_scalar=self.object_splitting_scalar,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
        )

        for filter in filters:
            for r_prop in filter["reqs_props"]:
                reqs["props"][r_prop] = ComputeObjectScale(
                    base_name=self.base_name,
                    variable=r_prop,
                    object_splitting_scalar=self.object_splitting_scalar,
                    mask_method=self.mask_method,
                    mask_method_extra_args=self.mask_method_extra_args,
                )

            if len(filters["extra"]) > 0:
                for r_field in filter["extra"]:
                    if r_field.startswith("tracked_"):
                        _, tracking_type = r_field.split("_")
                        reqs["extra"][r_field].append(
                            PerformObjectTracking2D(
                                base_name=self.base_name, tracking_type=tracking_type
                            )
                        )
                    else:
                        raise NotImplementedError(r_field)
        return reqs

    def _parse_filter_defs(self):
        filters = dict(reqd_props=[], fns=[])
        s_filters = sorted(self.filter_defs.split(","))
        for s_filter in s_filters:
            try:
                f_type, f_cond = s_filter.split(":")
                if f_type == "prop":
                    s_prop_and_op, s_value = f_cond.split("=")
                    i = s_prop_and_op.rfind("__")
                    prop_name, op_name = s_prop_and_op[:i], s_prop_and_op[i + 2 :]
                    op = dict(lt="less_than", gt="greater_than", eq="equals")[op_name]
                    value = float(s_value)
                    fn_base = objects.filter.filter_objects_by_property
                    fn = partial(fn_base, op=op, value=value)

                    filters["reqd_props"].append(prop_name)
                    filters["fns"].append(fn)
                else:
                    raise NotImplementedError(
                        "Filter type `{}` not recognised" "".format(f_type)
                    )
            except (IndexError, ValueError) as e:
                raise Exception(
                    "Malformed filter definition: `{}` {}".format(s_filter, e)
                )
        return filters

    def run(self):
        input = self.input()
        da_obj = input["objects"].open()

        filters = self._parse_filter_defs()

        for object_filter in filters:
            fn = object_filter["fn"]
            kws = dict(objects=da_obj)

            if "reqd_props" in object_filter:
                raise NotImplementedError()
            if "extra" in object_filter:
                for extra_req in object_filter["extra"]:
                    assert extra_req == "tracked_thermals"
                    kws["cloud_data"] = input["extra"][extra_req]

            da_obj = fn(**kws)

        da_obj.to_netcdf(self.output().fn)

    def output(self):
        mask_name = MakeMask.make_mask_name(
            base_name=self.base_name,
            method_name=self.mask_method,
            method_extra_args=self.mask_method_extra_args,
        )
        objects_name = objects.identify.make_objects_name(
            mask_name=mask_name, splitting_var=self.object_splitting_scalar
        )
        s_filters = self.filter_defs.replace(",", ".").replace("=", "")
        objects_name = "{}.filtered_by.{}".format(objects_name, s_filters)
        fn = objects.identify.OUT_FILENAME_FORMAT.format(
            base_name=self.base_name, objects_name=objects_name
        )
        p = get_workdir() / self.base_name / fn
        return XArrayTarget(str(p))


class ComputeObjectMinkowskiScales(luigi.Task):
    object_splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    object_filters = luigi.Parameter(default=None)

    def requires(self):
        return IdentifyObjects(
            base_name=self.base_name,
            splitting_scalar=self.object_splitting_scalar,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            filters=self.object_filters,
        )

    def run(self):
        da_objects = xr.open_dataarray(self.input().fn)

        ds = objects.minkowski_scales.main(da_objects=da_objects)

        ds.to_netcdf(self.output().fn)

    def output(self):
        if not self.input().exists():
            return luigi.LocalTarget("fakefile.nc")

        da_objects = xr.open_dataarray(self.input().fn, decode_times=False)
        objects_name = da_objects.name

        fn = objects.minkowski_scales.FN_FORMAT.format(
            base_name=self.base_name, objects_name=objects_name
        )

        p = get_workdir() / self.base_name / fn
        return XArrayTarget(str(p))


class ComputeObjectScale(luigi.Task):
    object_splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")

    variable = luigi.Parameter()

    def requires(self):
        reqs = {}

        reqs["objects"] = IdentifyObjects(
            base_name=self.base_name,
            splitting_scalar=self.object_splitting_scalar,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
        )

        required_fields = objects.integrate.get_integration_requirements(
            variable=self.variable
        )

        if "__" in self.variable:
            v, _ = self._get_var_and_op()
            required_fields[v] = v

        for k, v in required_fields.items():
            reqs[k] = ExtractField3D(base_name=self.base_name, field_name=v)

        return reqs

    def _get_var_and_op(self):
        if "__" in self.variable:
            return self.variable.split("__")
        else:
            return self.variable, None

    def output(self):
        objects_name = IdentifyObjects.make_name(
            base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            filter_defs=None,
        )

        variable, operator = self._get_var_and_op()
        name = objects.integrate.make_name(variable=variable, operator=operator)

        fn = objects.integrate.FN_OUT_FORMAT.format(
            base_name=self.base_name, objects_name=objects_name, name=name
        )
        p = get_workdir() / self.base_name / fn
        return XArrayTarget(str(p))

    def run(self):
        inputs = self.input()
        da_objects = xr.open_dataarray(inputs.pop("objects").fn)
        kwargs = dict((k, v.open()) for (k, v) in inputs.items())

        variable, operator = self._get_var_and_op()
        ds = objects.integrate.integrate(
            objects=da_objects, variable=variable, operator=operator, **kwargs
        )

        ds.to_netcdf(self.output().fn)


class ComputePerObjectAtHeight(luigi.Task):
    """
    For each object defined by `mask_method`, `mask_method_extra_args` and
    `object_splitting_scalar` compute the operation of `op` applied
    on `field_name` at height `z`
    """

    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    object_splitting_scalar = luigi.Parameter()

    field_name = luigi.OptionalParameter(default=None)
    op = luigi.Parameter()
    z = luigi.FloatParameter()

    def requires(self):
        tasks = dict(
            objects=IdentifyObjects(
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                splitting_scalar=self.object_splitting_scalar,
            ),
        )

        if self.op != "num_cells":
            tasks["field"] = ExtractField3D(
                base_name=self.base_name,
                field_name=self.field_name,
            )
        return tasks

    def run(self):
        # for `num_cells` operation the field shouldn't be given because the
        # number of cells is just computed from the mask
        inputs = self.input()

        da_objects = inputs["objects"].open()
        if self.op != "num_cells":
            da_field = inputs["field"].open().squeeze()
        else:
            if self.field_name is not None:
                raise Exception(
                    f"Field name should not be given when computing `{self.op}`"
                    f" (`{self.field_name}` was provided)"
                )
            da_field = None

        object_ids = np.unique(da_objects.chunk(None).values)
        if object_ids[0] == 0:
            object_ids = object_ids[1:]

        kwargs = dict(
            objects=da_objects.name,
            object_ids=object_ids,
            op=self.op,
        )

        if self.op != "num_cells":
            kwargs["scalar"] = da_field.name

        da_objects_ = da_objects.sel(zt=self.z).compute()
        if self.op != "num_cells":
            da_ = da_field.sel(zt=self.z).compute()
        else:
            da_ = xr.ones_like(da_objects_)

        # to avoid the confusion where the "area" is requested but what in fact
        # is returned is the "number of cells" (which is dimensionless) we
        # enforce here that the "area" cannot be calculated, but instead
        # "num_cells" can be requested and we use the `area` dask-image op
        # (which returns the number of cells)
        op = kwargs["op"]
        if op == "area":
            raise Exception(
                "Shouldn't ask for `area` as it asctually the number of cells"
            )
        elif op == "num_cells":
            op = "area"

        fn = getattr(dask_image.ndmeasure, op)
        v = fn(da_, label_image=da_objects_, index=object_ids).compute()
        da = xr.DataArray(data=v, dims=["object_id"], coords=dict(object_id=object_ids))
        if self.op != "num_cells":
            da.name = "{}__{}".format(da_.name, kwargs["op"])
            da.attrs["units"] = da_.units
            da.attrs["long_name"] = "{} of {} per object".format(
                kwargs["op"],
                da_.long_name,
            )
        else:
            da.name = "num_cells"
            da.attrs["units"] = "1"
            da.attrs["long_name"] = "num_cells per object"

        da.coords["zt"] = self.z
        da.coords["time"] = da_objects_.time

        da.to_netcdf(self.output().fn)

    def output(self):
        mask_name = MakeMask.make_mask_name(
            base_name=self.base_name,
            method_name=self.mask_method,
            method_extra_args=self.mask_method_extra_args,
        )
        fn = (
            f"{self.base_name}.{mask_name}.{self.field_name}__{self.op}_at_z{self.z}.nc"
        )
        p = get_workdir() / self.base_name / fn
        target = XArrayTarget(str(p))
        return target


class ComputePerObjectProfiles(luigi.Task):
    """
    For each object defined by `mask_method`, `mask_method_extra_args` and
    `object_splitting_scalar` compute a profile of the operation `op` applied
    on `field_name` as a function of height
    """

    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    object_splitting_scalar = luigi.Parameter()

    field_name = luigi.OptionalParameter(default=None)
    op = luigi.Parameter()
    z_max = luigi.FloatParameter(default=None)

    def requires(self):
        return IdentifyObjects(
            base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            splitting_scalar=self.object_splitting_scalar,
        )

    def run(self):
        da_objects = self.input().open()

        z_values = da_objects.sel(zt=slice(None, self.z_max)).zt.values

        tasks = [
            ComputePerObjectAtHeight(
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                object_splitting_scalar=self.object_splitting_scalar,
                field_name=self.field_name,
                op=self.op,
                z=z,
            )
            for z in z_values
        ]

        outputs = yield tasks

        da_by_height = xr.concat([output.open() for output in outputs], dim="zt")
        da_by_height.to_netcdf(self.output().fn)

    def output(self):
        mask_name = MakeMask.make_mask_name(
            base_name=self.base_name,
            method_name=self.mask_method,
            method_extra_args=self.mask_method_extra_args,
        )
        fn = (
            "{base_name}.{mask_name}.{field_name}__{op}"
            ".by_z_per_object{ex}.nc".format(
                base_name=self.base_name,
                mask_name=mask_name,
                field_name=self.field_name,
                op=self.op,
                ex=self.z_max is None and "" or "_to_z" + str(self.z_max),
            )
        )
        p = get_workdir() / self.base_name / fn
        target = XArrayTarget(str(p))
        return target


class ComputeFieldDecompositionByHeightAndObjects(luigi.Task):
    field_name = luigi.Parameter()

    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    object_splitting_scalar = luigi.Parameter()

    object_filters = luigi.Parameter(default=None)
    z_max = luigi.FloatParameter(default=None)

    def requires(self):
        tasks = dict(
            decomp_profile_ncells=ComputePerObjectProfiles(
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                object_splitting_scalar=self.object_splitting_scalar,
                op="num_cells",
                z_max=self.z_max,
            ),
            decomp_profile_sum=ComputePerObjectProfiles(
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                object_splitting_scalar=self.object_splitting_scalar,
                field_name=self.field_name,
                op="sum",
                z_max=self.z_max,
            ),
            da_3d=ExtractField3D(
                base_name=self.base_name,
                field_name=self.field_name,
            ),
            mask=MakeMask(
                base_name=self.base_name,
                method_name=self.mask_method,
                method_extra_args=self.mask_method_extra_args,
            ),
        )

        if self.object_filters is not None:
            # if we're filtering on a set of scales we just compute a scale
            # here that's easy so that we get the objects which satisfy the
            # filter
            tasks["scales"] = ComputeObjectScales(
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                object_splitting_scalar=self.object_splitting_scalar,
                variables="num_cells",
                object_filters=self.object_filters,
            )
        return tasks

    def _run_single(self):
        input = self.input()
        da_field_ncells_per_object = input["decomp_profile_ncells"].open().squeeze()
        da_field_sum_per_object = input["decomp_profile_sum"].open()
        da_3d = input["da_3d"].open().squeeze()
        da_mask = input["mask"].open().squeeze()
        nx, ny = da_3d.xt.count(), da_3d.yt.count()

        # calculate domain mean profile
        da_domain_mean_profile = da_3d.mean(
            dim=("xt", "yt"), dtype=np.float64, skipna=True
        )
        da_domain_mean_profile["sampling"] = "full domain"

        # # contributions from mask only
        # calculate mask mean profile and mask fractional area (so that
        # total contribution to domain mean can be computed later)
        da_mask_mean_profile = da_3d.where(da_mask).mean(
            dim=("xt", "yt"), dtype=np.float64, skipna=True
        )
        da_mask_mean_profile["sampling"] = "mask"
        da_mask_areafrac_profile = da_mask.sum(
            dim=("xt", "yt"), dtype=np.float64, skipna=True
        ) / (nx * ny)
        da_mask_areafrac_profile["sampling"] = "mask"

        # # contributions from objects
        # if object filters have been provided we should only include the
        # objects which are in the filtered scales file (as these satisfy the
        # filtering criteria)
        if self.object_filters is not None:
            ds_scales = input["scales"].open()
            # need to cast scales indexing (int64) to object identifitcation
            # indexing (uint32) here, otherwise saving goes wrong when merging
            # (because xarray makes the dtype `object` otherwise)
            ds_scales["object_id"] = ds_scales.object_id.astype(
                da_field_ncells_per_object.object_id.dtype
            )

            def filter_per_object_field(da_field):
                return da_field.where(da_field.object_id == ds_scales.object_id)

            da_field_ncells_per_object = filter_per_object_field(
                da_field=da_field_ncells_per_object
            )
            da_field_sum_per_object = filter_per_object_field(
                da_field=da_field_sum_per_object
            )

        # calculate objects mean profile and objects fractional area (so
        # that total contribution to domain mean can be computed later)
        da_objects_total_flux = da_field_sum_per_object.sum(
            dim=("object_id",), dtype=np.float64, skipna=True
        )
        da_objects_total_ncells = da_field_ncells_per_object.sum(
            dim=("object_id",), dtype=np.float64, skipna=True
        )
        da_objects_mean_profile = da_objects_total_flux / da_objects_total_ncells
        da_objects_mean_profile["sampling"] = "objects"
        da_objects_areafrac_profile = da_objects_total_ncells / (nx * ny)
        da_objects_areafrac_profile["sampling"] = "objects"

        da_mean_profiles = xr.concat(
            [da_domain_mean_profile, da_mask_mean_profile, da_objects_mean_profile],
            dim="sampling",
        )
        da_mean_profiles.name = "{}__mean".format(self.field_name)
        da_mean_profiles.attrs = dict(
            units=da_3d.units, long_name="{} mean".format(da_3d.long_name)
        )

        da_areafrac_profiles = xr.concat(
            [da_mask_areafrac_profile, da_objects_areafrac_profile], dim="sampling"
        )
        da_areafrac_profiles.name = "areafrac"
        da_areafrac_profiles.attrs = dict(units="1", long_name="area fraction")

        ds_profiles = xr.merge([da_mean_profiles, da_areafrac_profiles])

        ds = xr.merge(
            [
                ds_profiles,
                da_field_ncells_per_object,
                da_field_sum_per_object,
            ]
        )

        if self.z_max is not None:
            ds = ds.sel(zt=slice(None, self.z_max))

        ds.attrs["nx"] = int(nx)
        ds.attrs["ny"] = int(ny)
        return ds

    def run(self):
        if "+" in self.base_name:
            ds = self._run_multiple()
        else:
            ds = self._run_single()

        fn = self.output().fn
        Path(fn).parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(fn)

    def output(self):
        mask_name = MakeMask.make_mask_name(
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
            "{base_name}.{mask_name}.{field_name}"
            "{s_filter}.by_z_per_object{ex}.{filetype}".format(
                base_name=self.base_name,
                mask_name=mask_name,
                field_name=self.field_name,
                filetype="nc",
                s_filter=s_filter,
                ex=self.z_max is None and "" or "_to_z" + str(self.z_max),
            )
        )

        p = get_workdir() / self.base_name / fn
        target = XArrayTarget(str(p))
        return target


class EstimateCharacteristicScales(luigi.Task):
    object_splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    variables = ["length", "width", "thickness"]
    object_filters = luigi.Parameter(default=None)
    fit_type = luigi.Parameter(default="exponential")

    def requires(self):
        return ComputeObjectScales(
            variables=",".join(self.variables),
            base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            object_filters=self.object_filters,
        )

    def run(self):
        ds = self.input().open()
        assert self.fit_type == "exponential"
        fn = length_scales.model_fitting.exponential_fit.fit
        ds_scales = ds[self.variables].apply(fn)
        ds_scales.to_netcdf(self.output().fn)

    def output(self):
        mask_name = MakeMask.make_mask_name(
            base_name=self.base_name,
            method_name=self.mask_method,
            method_extra_args=self.mask_method_extra_args,
        )
        fn = "{}.{}.exp_fit_scales.nc".format(self.base_name, mask_name)
        p = get_workdir() / self.base_name / fn
        target = XArrayTarget(str(p))
        return target


class MakeMaskWithObjects(MakeMask):
    filtered_by = luigi.Parameter()
    object_splitting_scalar = luigi.Parameter()
    object_filters = luigi.Parameter()

    def requires(self):
        reqs = {}
        is_filtered = "__filtered_by=" in self.method_name
        if is_filtered:
            method_name, filters = self.method_name.split("__filtered_by=")
        else:
            method_name = self.method_name

        object_splitting_scalar = self.object_splitting_scalar
        method_extra_args = self.method_extra_args

        reqs["all_objects"] = IdentifyObjects(
            base_name=self.base_name,
            splitting_scalar=object_splitting_scalar,
            mask_method=method_name,
            mask_method_extra_args=method_extra_args,
        )

        if "tracking" in filters:
            raise NotImplementedError
            assert filters == "tracking:triggers_cloud"
            reqs["tracking"] = PerformObjectTracking2D(
                base_name=self.base_name,
                tracking_type=objects.filter.TrackingType.THERMALS_ONLY,
            )

        else:

            reqs["filtered_objects"] = ComputeObjectScales(
                base_name=self.base_name,
                variables="num_cells",
                mask_method=method_name,
                mask_method_extra_args=method_extra_args,
                object_splitting_scalar=object_splitting_scalar,
                object_filters=filters,
            )

        return reqs

    def run(self):
        raise NotImplementedError
        # object_splitting_scalar = self.object_splitting_scalar
        # method_extra_args = self.method_extra_args
        method_name = self.method_name

        mask_functions = None
        cloud_identification = None
        object_filters = None

        method_kwargs = self._build_method_kwargs(
            base_name=self.base_name, method_extra_args=self.method_extra_args
        )
        mask_fn = getattr(mask_functions, method_name)
        assert hasattr(mask_fn, "description")

        input = self.input()
        da_objects = input["all_objects"].open(decode_times=False)

        if "tracking:" in self.method_name:
            raise NotImplementedError
            cloud_data = self.requires()["tracking"].get_cloud_data()

            t0 = da_objects.time.values

            ds_track_2d = cloud_data._fh_track.sel(time=t0)
            objects_tracked_2d = ds_track_2d.nrthrm

            da_mask = da_objects.where(~objects_tracked_2d.isnull())
            filter_desc = "cloud_trigger"
        else:
            ds_obj_props_filtered = input["filtered_objects"].open()

            labels = da_objects.values

            cloud_identification.filter_labels(
                labels=labels, idxs_keep=ds_obj_props_filtered.object_id.values
            )

            da_mask = xr.DataArray(
                labels != 0, coords=da_objects.coords, dims=da_objects.dims
            )

            filter_desc = objects.filter.latex_format(object_filters)

        mask_desc = mask_fn.description.format(**method_kwargs)
        da_mask.attrs["long_name"] = "{} filtered by {}".format(mask_desc, filter_desc)

        da_mask.name = self.method_name
        da_mask.to_netcdf(self.output().fn)

    @classmethod
    def make_mask_name(cls, base_name, method_name, method_extra_args, object_filters):
        mask_name = super().make_mask_name(
            base_name=base_name,
            method_name=method_name,
            method_extra_args=method_extra_args,
        )

        s_filters = object_filters.replace(",", ".").replace("=", "").replace(":", "__")
        mask_name += ".filtered_by." + s_filters
        if mask_name.endswith("."):
            mask_name = mask_name[:-1]

        return mask_name

    def output(self):
        mask_name = self.make_mask_name(
            base_name=self.base_name,
            method_name=self.method_name,
            method_extra_args=self.method_extra_args,
            object_filters=self.object_filters,
        )

        fn = make_mask.OUT_FILENAME_FORMAT.format(
            base_name=self.base_name, mask_name=mask_name
        )
        p = get_workdir() / self.base_name / fn
        return XArrayTarget(str(p))


class ObjectTwoScalesComposition(luigi.Task):
    """
    Decompose `field_name` by height and object property `x`, with number and
    mean-flux distributions shown as margin plots. An extract reference profile
    can by defining the filters on the objects to consider with
    `ref_profile_object_filters`
    """

    x = luigi.Parameter()
    dx = luigi.FloatParameter(default=None)

    y = luigi.Parameter()
    dy = luigi.FloatParameter(default=None)

    z = luigi.Parameter()
    field_name = luigi.Parameter()

    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default="")
    object_splitting_scalar = luigi.Parameter()
    z_max = luigi.FloatParameter(default=None)

    method = luigi.Parameter()

    object_filters = luigi.Parameter(default=None)

    def requires(self):
        kwargs = dict(
            base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            field_name=self.field_name,
            z_max=self.z_max,
        )

        reqs = dict(
            base=ComputeFieldDecompositionByHeightAndObjects(
                object_filters=self.object_filters,
                **kwargs,
            ),
            scales=ComputeObjectScales(
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                object_splitting_scalar=self.object_splitting_scalar,
                variables=f"{self.x},{self.y}",
                object_filters=self.object_filters,
            ),
            field=ExtractField3D(base_name=self.base_name, field_name=self.field_name),
        )

        return reqs

    def run(self):
        da3d_field = self.input()["field"].open()
        ds = self.input()["base"].open()

        # make the scale we're binning on available
        ds_scales = self.input()["scales"].open()
        ds = xr.merge([ds, ds_scales])

        nx = int(da3d_field.xt.count())
        ny = int(da3d_field.yt.count())
        domain_num_cells = nx * ny

        da = objects.flux_contribution.make_2d_decomposition(
            ds=ds,
            x=self.x,
            y=self.y,
            v=self.field_name,
            z=self.z,
            dx=self.dx,
            dy=self.dy,
            domain_num_cells=domain_num_cells,
            method=self.method,
        )

        da.to_netcdf(self.output().fn)

    def output(self):
        mask_name = MakeMask.make_mask_name(
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
            "{base_name}.{mask_name}.{field_name}__by__{x}{dx}_and_{y}{dy}"
            "{method}.{s_filter}.{filetype}".format(
                base_name=self.base_name,
                mask_name=mask_name,
                field_name=self.field_name,
                x=self.x,
                dx=self.dx or "",
                y=self.y,
                dy=self.dy or "",
                method=self.method,
                filetype="nc",
                s_filter=s_filter,
            )
        )
        p = get_workdir() / self.base_name / fn
        return XArrayTarget(str(p))
