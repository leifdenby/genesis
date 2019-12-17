import os
import subprocess
from pathlib import Path
import re
import warnings
from functools import partial
import hashlib

import ipdb
import luigi
import xarray as xr
import numpy as np
import yaml
import hues
from tqdm import tqdm

from .. import mask_functions, make_mask
from ... import objects
from ...bulk_statistics import cross_correlation_with_height
from ...utils import find_vertical_grid_spacing, calc_flux
from ...objects import property_filters
from ...objects import integral_properties
from ... import length_scales

import cloud_identification

import dask_image

if 'USE_SCHEDULER' in os.environ:
    from dask.distributed import Client
    client = Client(threads_per_worker=1)

import importlib

try:
    import cloud_tracking_analysis.cloud_data
    from cloud_tracking_analysis import CloudData, CloudType, cloud_operations
    HAS_CLOUD_TRACKING = True
except ImportError:
    HAS_CLOUD_TRACKING = False

DATA_SOURCES = None
WORKDIR = Path("data")

def add_datasource(name, attrs):
    global DATA_SOURCES
    if DATA_SOURCES is None:
        DATA_SOURCES = {}
    DATA_SOURCES[name] = attrs

def set_workdir(path):
    global WORKDIR
    WORKDIR = Path(path)

def get_datasources():
    if DATA_SOURCES is not None:
        datasources = DATA_SOURCES
    else:
        try:
            with open('datasources.yaml') as fh:
                loader = getattr(yaml, 'FullLoader', yaml.Loader)
                datasources = yaml.load(fh, Loader=loader)
        except IOError:
            raise Exception("please define your data sources in datasources.yaml")

    return datasources

def _get_dataset_meta_info(base_name):
    datasources = get_datasources()

    datasource = None
    if datasources is not None:
        if base_name in datasources:
            datasource = datasources[base_name]
        elif re.search(r"\.tn\d+$", base_name):
            base_name, timestep = base_name.split('.tn')
            datasource = datasources[base_name]
            datasource["timestep"] = int(timestep)

    if datasource is None:
        raise Exception("Please make a definition for `{}` in "
                        "datasources.yaml".format(base_name))

    return datasource


class XArrayTarget(luigi.target.FileSystemTarget):
    fs = luigi.local_target.LocalFileSystem()

    def __init__(self, path, *args, **kwargs):
        super(XArrayTarget, self).__init__(path, *args, **kwargs)
        self.path = path

    def open(self, *args, **kwargs):
        # ds = xr.open_dataset(self.path, engine='h5netcdf', *args, **kwargs)
        ds = xr.open_dataset(self.path, *args, **kwargs)

        if len(ds.data_vars) == 1:
            name = list(ds.data_vars)[0]
            da = ds[name]
            da.name = name
            return da
        else:
            return ds

    @property
    def fn(self):
        return self.path

COMPOSITE_FIELD_METHODS = dict(
    p_stddivs=(mask_functions.calc_scalar_perturbation_in_std_div, []),
    flux=(calc_flux.compute_vertical_flux, ['w',]),
    _prefix__d=(calc_flux.get_horz_devition, []),
)

class ExtractField3D(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    FN_FORMAT = "{experiment_name}.{field_name}.nc"

    @staticmethod
    def _get_data_loader_module(meta):
        model_name = meta.get('model')
        if model_name is None:
            model_name = 'UCLALES'

        module_name = ".data_sources.{}".format(
            model_name.lower().replace('-', '_')
        )
        return importlib.import_module(module_name,
                                       package='genesis.utils.pipeline')

    def requires(self):
        meta = _get_dataset_meta_info(self.base_name)
        data_loader = self._get_data_loader_module(meta=meta)

        reqs = {}

        derived_fields = getattr(data_loader, 'DERIVED_FIELDS', None)

        if derived_fields is not None:
            for req_field in derived_fields.get(self.field_name, []):
                reqs[req_field] = ExtractField3D(base_name=self.base_name,
                                                 field_name=req_field)

        for (affix, (func, extra_fields)) in COMPOSITE_FIELD_METHODS.items():
            req_field = None
            if affix.startswith('_prefix__'):
                prefix = affix.replace('_prefix__', '')
                if self.field_name.startswith(prefix):
                    req_field = self.field_name.replace('{}_'.format(prefix), '')
            else:
                postfix = affix
                if self.field_name.endswith(postfix):
                    req_field = self.field_name.replace('_{}'.format(postfix), '')

            if req_field is not None:
                reqs['da'] = ExtractField3D(base_name=self.base_name,
                                                 field_name=req_field)
                for v in extra_fields:
                    reqs[v] = ExtractField3D(base_name=self.base_name,
                                             field_name=v)

        return reqs

    def run(self):
        meta = _get_dataset_meta_info(self.base_name)

        fn_out = self.output()

        if fn_out.exists():
            pass
        elif meta['host'] == 'localhost':
            p_out = Path(self.output().fn)
            p_out.parent.mkdir(parents=True, exist_ok=True)

            is_composite = False
            for (affix, (func, _)) in COMPOSITE_FIELD_METHODS.items():
                if affix.startswith('_prefix__'):
                    prefix = affix.replace('_prefix__', '')
                    is_composite = self.field_name.startswith(prefix)
                else:
                    postfix = affix
                    is_composite = self.field_name.endswith(postfix)

                if is_composite:
                    das_input = dict([
                        (k, input.open(decode_times=False))
                        for (k, input) in self.input().items()
                    ])
                    with ipdb.launch_ipdb_on_exception():
                        da = func(**das_input)
                    # XXX: remove infs for now
                    da = da.where(~np.isinf(da))
                    da.to_netcdf(self.output().fn)
                    break

            if not is_composite:
                opened_inputs = dict([
                    (k, input.open()) for (k, input) in self.input().items()
                ])
                data_loader = self._get_data_loader_module(meta=meta)
                data_loader.extract_field_to_filename(
                    dataset_meta=meta, path_out=p_out,
                    field_name=self.field_name,
                    **opened_inputs
                )
        else:
            raise NotImplementedError(fn_out.fn)

    def output(self):
        meta = _get_dataset_meta_info(self.base_name)

        fn = self.FN_FORMAT.format(
            experiment_name=meta['experiment_name'], timestep=meta['timestep'],
            field_name=self.field_name
        )

        p = WORKDIR/self.base_name/fn

        t = XArrayTarget(str(p))

        if t.exists():
            data = t.open()
            if isinstance(data, xr.Dataset):
                if len(data.variables) == 0:
                    warnings.warn("Stored file for `{}` is empty, deleting..."
                                  "".format(self.field_name))
                    p.unlink()

        return t


class MakeMask(luigi.Task):
    base_name = luigi.Parameter()
    method_extra_args = luigi.Parameter(default='')
    method_name = luigi.Parameter()

    def requires(self):

        reqs = {}
        is_filtered = "__filtered_by=" in self.method_name
        if is_filtered:
            method_name, object_filters = self.method_name.split('__filtered_by=')
        else:
            method_name = self.method_name

        if is_filtered:

            if not "object_splitting_scalar=" in self.method_extra_args:
                raise Exception("You must provide the object splitting scalar"
                                " when creating a mask from filtered object"
                                " properties (with --mask-method-extra-args)")
            else:
                extra_args = []
                for item in self.method_extra_args.split(','):
                    k, v = item.split('=')
                    if k == 'object_splitting_scalar':
                        object_splitting_scalar = v
                    else:
                        extra_args.append(item)
                method_extra_args = ','.join(extra_args)

            reqs['filtered_objects'] = ComputeObjectScales(
                base_name=self.base_name,
                variables="num_cells",
                mask_method=method_name,
                mask_method_extra_args=method_extra_args,
                object_splitting_scalar=object_splitting_scalar,
                object_filters=object_filters,
            )

            reqs['all_objects'] = IdentifyObjects(
                base_name=self.base_name,
                splitting_scalar=object_splitting_scalar,
                mask_method=method_name,
                mask_method_extra_args=method_extra_args,
            )
        else:

            method_kwargs = self._build_method_kwargs(
                base_name=self.base_name, method_extra_args=self.method_extra_args
            )

            try:
                make_mask.build_method_kwargs(method=method_name, kwargs=method_kwargs)
            except make_mask.MissingInputException as e:
                for v in e.missing_kwargs:
                    reqs[v] = ExtractField3D(field_name=v, base_name=self.base_name)

        return reqs

    @staticmethod
    def _build_method_kwargs(base_name, method_extra_args):
        kwargs = dict(base_name=base_name)
        for kv in method_extra_args.split(","):
            if kv == "":
                continue
            elif kv.startswith("object_splitting_scalar="):
                continue
            k,v = kv.split("=")
            kwargs[k] = v
        return kwargs

    def run(self):
        if 'filtered_objects' in self.input():
            method_name, object_filters = self.method_name.split('__filtered_by=')
            method_kwargs = self._build_method_kwargs(
                base_name=self.base_name, method_extra_args=self.method_extra_args
            )
            mask_fn = getattr(mask_functions, method_name)
            assert hasattr(mask_fn, "description")

            input = self.input()
            ds_obj_props_filtered = input['filtered_objects'].open()
            da_objects = input['all_objects'].open()

            labels = da_objects.values

            cloud_identification.filter_labels(
                labels=labels,
                idxs_keep=ds_obj_props_filtered.object_id.values
            )

            da_mask = xr.DataArray(
                labels != 0, coords=da_objects.coords,
                dims=da_objects.dims
            )

            mask_desc = mask_fn.description.format(**method_kwargs)
            filter_desc = objects.property_filters.latex_format(object_filters)
            da_mask.attrs['long_name'] = "{} filtered by {}".format(
                mask_desc, filter_desc
            )

            da_mask.name = self.method_name
            da_mask.to_netcdf(self.output().fn)
        else:
            method_kwargs = self._build_method_kwargs(
                base_name=self.base_name, method_extra_args=self.method_extra_args
            )

            for (v, input) in self.input().items():
                method_kwargs[v] = xr.open_dataarray(input.fn, decode_times=False)

            cwd = os.getcwd()
            p_data = WORKDIR/self.base_name
            os.chdir(p_data)
            mask = make_mask.main(method=self.method_name, method_kwargs=method_kwargs)
            os.chdir(cwd)
            mask.to_netcdf(self.output().fn)

    @classmethod
    def make_mask_name(cls, base_name, method_name, method_extra_args):
        kwargs = cls._build_method_kwargs(
            base_name=base_name, method_extra_args=method_extra_args
        )

        is_filtered = "__filtered_by=" in method_name
        if is_filtered:
            method_name, object_filters = method_name.split('__filtered_by=')

        try:
            kwargs = make_mask.build_method_kwargs(
                method=method_name, kwargs=kwargs
            )
        except make_mask.MissingInputException as e:
            for v in e.missing_kwargs:
                kwargs[v] = None
        kwargs = make_mask.build_method_kwargs(
            method=method_name, kwargs=kwargs
        )

        mask_name = make_mask.make_mask_name(
            method=method_name, method_kwargs=kwargs
        )

        if is_filtered:
            s_filters = (object_filters.replace(',','.')
                                       .replace('=', '')
                                       .replace(':', '__'))
            mask_name += ".filtered_by." + s_filters
            if mask_name.endswith('.'):
                mask_name = mask_name[:-1]

        return mask_name

    def output(self):
        mask_name = self.make_mask_name(
            base_name=self.base_name,
            method_name=self.method_name,
            method_extra_args=self.method_extra_args
        )

        fn = make_mask.OUT_FILENAME_FORMAT.format(
            base_name=self.base_name, mask_name=mask_name
        )
        p = WORKDIR/self.base_name/fn
        return XArrayTarget(str(p))

class FilterObjects(luigi.Task):
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')
    object_splitting_scalar = luigi.Parameter()
    filter_defs = luigi.Parameter()

    def requires(self):
        filters = self._parse_filter_defs()
        reqs = {}
        reqs['objects'] = IdentifyObjects(
            base_name=self.base_name,
            splitting_scalar=self.object_splitting_scalar,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
        )

        reqs['props'] = [
            ComputeObjectScale(
                base_name=self.base_name,
                variable=reqd_prop,
                object_splitting_scalar=self.object_splitting_scalar,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
            )
            for reqd_prop in filters['reqd_props']
        ]
        return reqs

    def _parse_filter_defs(self):
        filters = dict(reqd_props=[], fns=[])
        s_filters = sorted(self.filter_defs.split(','))
        for s_filter in s_filters:
            try:
                f_type, f_cond = s_filter.split(':')
                if f_type == 'prop':
                    s_prop_and_op, s_value = f_cond.split("=")
                    i = s_prop_and_op.rfind('__')
                    prop_name, op_name = s_prop_and_op[:i], s_prop_and_op[i+2:]
                    op = dict(lt="less_than", gt="greater_than", eq="equals")[op_name]
                    value = float(s_value)
                    fn_base = objects.filter.filter_objects_by_property
                    fn = partial(fn_base, op=op, value=value)

                    filters['reqd_props'].append(prop_name)
                    filters['fns'].append(fn)
                else:
                    raise NotImplementedError("Filter type `{}` not recognised"
                                              "".format(f_type))
            except (IndexError, ValueError) as e:
                raise Exception("Malformed filter definition: `{}`".format(
                                s_filter))
        return filters

    def run(self):
        input = self.input()
        da_obj = input['objects'].open()

        filters = self._parse_filter_defs()

        for fn, prop in zip(filters['fns'], input['props']):
            da_obj = fn(objects=da_obj, da_property=prop.open())

        da_obj.to_netcdf(self.output().fn)

    def output(self):
        mask_name = MakeMask.make_mask_name(
            base_name=self.base_name,
            method_name=self.mask_method,
            method_extra_args=self.mask_method_extra_args
        )
        objects_name = objects.identify.make_objects_name(
            mask_name=mask_name,
            splitting_var=self.object_splitting_scalar
        )
        s_filters = self.filter_defs.replace(',','.').replace('=', '')
        objects_name = "{}.filtered_by.{}".format(objects_name, s_filters)
        fn = objects.identify.OUT_FILENAME_FORMAT.format(
            base_name=self.base_name, objects_name=objects_name
        )
        p = WORKDIR/self.base_name/fn
        return XArrayTarget(str(p))

class IdentifyObjects(luigi.Task):
    splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')
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
                    method_extra_args=self.mask_method_extra_args
                ),
                scalar=ExtractField3D(
                    base_name=self.base_name,
                    field_name=self.splitting_scalar
                )
            )

    def run(self):
        if self.filters is not None:
            pass
        else:
            da_mask = xr.open_dataarray(self.input()['mask'].fn).squeeze()
            da_scalar = xr.open_dataarray(self.input()['scalar'].fn).squeeze()

            object_labels = objects.identify.label_objects(
                mask=da_mask, splitting_scalar=da_scalar
            )

            object_labels.to_netcdf(self.output().fn)

    @staticmethod
    def make_name(base_name, mask_method, mask_method_extra_args,
                  object_splitting_scalar, filter_defs,):
        mask_name = MakeMask.make_mask_name(
            base_name=base_name,
            method_name=mask_method,
            method_extra_args=mask_method_extra_args
        )
        objects_name = objects.identify.make_objects_name(
            mask_name=mask_name,
            splitting_var=object_splitting_scalar
        )
        if filter_defs is not None:
            s_filters = (filter_defs.replace(',','.')
                                    .replace('=', '')
                                    .replace(':', '__'))
            objects_name = "{}.filtered_by.{}".format(objects_name, s_filters)
        return objects_name

    def output(self):
        if self.filters is not None:
            return self.input()

        objects_name = self.make_name(
            base_name=self.base_name, mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.splitting_scalar,
            filter_defs=self.filters,
        )

        fn = objects.identify.OUT_FILENAME_FORMAT.format(
            base_name=self.base_name, objects_name=objects_name
        )
        p = WORKDIR/self.base_name/fn

        return XArrayTarget(str(p))

class ComputeObjectMinkowskiScales(luigi.Task):
    object_splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')
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

        p = WORKDIR/self.base_name/fn
        return XArrayTarget(str(p))

class ComputeObjectScale(luigi.Task):
    object_splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')

    variable = luigi.Parameter()

    def requires(self):
        reqs = {}

        reqs['objects'] = IdentifyObjects(
            base_name=self.base_name,
            splitting_scalar=self.object_splitting_scalar,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
        )

        required_fields = objects.integrate.get_integration_requirements(
            variable=self.variable
        )

        if '__' in self.variable:
            v, _ = self._get_var_and_op()
            required_fields[v] = v

        for k, v in required_fields.items():
            reqs[k] = ExtractField3D(base_name=self.base_name, field_name=v)

        return reqs

    def _get_var_and_op(self):
        if '__' in self.variable:
            return self.variable.split('__')
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
        name = objects.integrate.make_name(variable=variable,
                                           operator=operator)

        fn = objects.integrate.FN_OUT_FORMAT.format(
            base_name=self.base_name, objects_name=objects_name,
            name=name
        )
        p = WORKDIR/self.base_name/fn
        return XArrayTarget(str(p))

    def run(self):
        inputs = self.input()
        da_objects = xr.open_dataarray(inputs.pop('objects').fn)
        kwargs = dict((k, v.open()) for (k, v) in inputs.items())

        variable, operator = self._get_var_and_op()
        ds = objects.integrate.integrate(objects=da_objects,
                                         variable=variable,
                                         operator=operator,
                                         **kwargs)
        ds.to_netcdf(self.output().fn)

def merge_object_datasets(dss):
    def _strip_coord(ds_):
        """
        remove the values of the `object_id` coordinate so that
        we can concate along it without having duplicates. We keep
        a copy of the original object id values
        """
        obj_id = ds_['object_id']
        del(ds_['object_id'])
        ds_['org_object_id'] = ('object_id'), obj_id.values
        return ds_
    dss = [_strip_coord(ds_) for ds_ in dss]
    ds = xr.concat(dss, dim="object_id")
    ds['object_id'] = np.arange(ds.object_id.max()+1)

    return ds


class ComputeObjectScales(luigi.Task):
    object_splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')
    variables = luigi.Parameter(default='com_angles')
    object_filters = luigi.Parameter(default=None)

    def requires(self):
        if "+" in self.base_name:
            # # "+" can be used a shorthand to concat objects together from
            # # different datasets
            base_names = self.base_name.split("+")
            reqs = dict([
                (base_name, ComputeObjectScales(
                                base_name=base_name,
                                mask_method=self.mask_method,
                                mask_method_extra_args=self.mask_method_extra_args,
                                variables=self.variables,
                                object_splitting_scalar=self.object_splitting_scalar,
                                object_filters=self.object_filters,
                            ))
                for base_name in base_names
            ])
            return reqs
        elif self.object_filters is not None:
            return FilterObjectScales(
                object_splitting_scalar=self.object_splitting_scalar,
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                variables=self.variables,
                object_filters=self.object_filters
            )
        else:
            variables = set(self.variables.split(','))
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
                if v == 'minkowski':
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
        if not "+" in self.base_name and self.object_filters is not None:
            pass
        else:
            if "+" in self.base_name:
                dss = [
                    fh.open(decode_times=False)
                    for (base_name, fh) in self.input().items()
                ]
                ds = merge_object_datasets(dss=dss)

                Path(self.output().fn).parent.mkdir(parents=True, exist_ok=True)
            else:
                ds = xr.merge([
                    input.open(decode_times=False) for input in self.input()
                ])

            objects_name = IdentifyObjects.make_name(
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                object_splitting_scalar=self.object_splitting_scalar,
                filter_defs=self.object_filters,
            )
            ds.attrs['base_name'] = self.base_name
            ds.attrs['objects_name'] = objects_name

            if isinstance(ds, xr.Dataset):
                ds = ds[self.variables.split(',')]
            else:
                assert ds.name == self.variables

            ds.to_netcdf(self.output().fn)

    def output(self):
        if not "+" in self.base_name and self.object_filters is not None:
            return self.input()

        objects_name = IdentifyObjects.make_name(
            base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            filter_defs=self.object_filters,
        )

        scales_identifier = hashlib.md5(self.variables.encode('utf-8')).hexdigest()
        fn = objects.integrate.FN_OUT_FORMAT.format(
            base_name=self.base_name, objects_name=objects_name,
            name="object_scales.{}".format(scales_identifier),
        )

        # XXX: filenames are max allowed to be 255 characters on linux, this is
        # a hack to generate a unique filename we can use
        if len(fn) > 255:
            fn = "{}.nc".format(hashlib.md5(fn.encode('utf-8')).hexdigest())

        p = WORKDIR/self.base_name/fn
        target = XArrayTarget(str(p))

        if target.exists():
            ds = target.open(decode_times=False)
            variables = self.variables.split(',')
            if isinstance(ds, xr.Dataset):
                if not set(ds.data_vars) == set(variables):
                    import ipdb
                    ipdb.set_trace()
            else:
                assert ds.name == variables[0]

        return target

class FilterObjectScales(ComputeObjectScales):
    object_splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')
    variables = luigi.Parameter(default='com_angles')
    object_filters = luigi.Parameter()

    def requires(self):
        filters = property_filters.parse_defs(self.object_filters)
        variables = self.variables.split(',')
        variables += filters['reqd_props']

        return ComputeObjectScales(
            variables=",".join(variables), base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            # no object filter, get properties for all objects
        )

    def run(self):
        input = self.input()
        ds_objs = input.open()
        if not isinstance(ds_objs, xr.Dataset):
            # if only one variable was requested we'll get a dataarray back
            ds_objs = ds_objs.to_dataset()

        filters = property_filters.parse_defs(self.object_filters)
        for fn in filters['fns']:
            ds_objs = fn(ds_objs)

        objects_name = IdentifyObjects.make_name(
            base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            filter_defs=self.object_filters,
        )
        ds_objs.attrs['base_name'] = self.base_name
        ds_objs.attrs['objects_name'] = objects_name

        ds_objs.to_netcdf(self.output().fn)

    def output(self):
        objects_name = IdentifyObjects.make_name(
            base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            filter_defs=self.object_filters,
        )

        fn = objects.integrate.FN_OUT_FORMAT.format(
            base_name=self.base_name, objects_name=objects_name,
            name="object_scales_collection"
        )

        p = WORKDIR/self.base_name/fn
        target = XArrayTarget(str(p))

        if target.exists():
            ds = target.open(decode_times=False)
            variables = self.variables.split(',')
            if isinstance(ds, xr.Dataset):
                if any([v not in ds.data_vars for v in variables]):
                    p.unlink()
            elif ds.name != self.variables:
                print(ds.name)
                p.unlink()

        return target

class ExtractCumulantScaleProfile(luigi.Task):
    base_name = luigi.Parameter()
    v1 = luigi.Parameter()
    v2 = luigi.Parameter()
    z_max = luigi.FloatParameter(default=700.)
    mask = luigi.Parameter(default=None)
    mask_args = luigi.Parameter(default='')
    width_method = length_scales.cumulant.calc.WidthEstimationMethod.MASS_WEIGHTED

    def requires(self):
        reqs = {}
        reqs['fields'] = [
                ExtractField3D(base_name=self.base_name,
                                    field_name=self.v1),
                ExtractField3D(base_name=self.base_name,
                                    field_name=self.v2),
        ]

        if self.mask is not None:
            reqs['mask'] = MakeMask(method_name=self.mask,
                                         method_extra_args=self.mask_args,
                                         base_name=self.base_name
                                         )

        return reqs

    def run(self):
        da_v1 = self.input()['fields'][0].open(decode_times=False)
        da_v2 = self.input()['fields'][1].open(decode_times=False)

        calc_fn = length_scales.cumulant.vertical_profile.calc.get_height_variation_of_characteristic_scales

        mask = None
        if self.mask:
            mask = self.input()['mask'].open(decode_times=False)


        import ipdb
        with ipdb.launch_ipdb_on_exception():
            da = calc_fn(
                v1_3d=da_v1, v2_3d=da_v2, width_method=self.width_method,
                z_max=self.z_max, mask=mask
            )

        da.to_netcdf(self.output().path)

    def output(self):
        fn = length_scales.cumulant.vertical_profile.calc.FN_FORMAT.format(
            base_name=self.base_name, v1=self.v1, v2=self.v2,
            mask=self.mask or "no_mask"
        )
        p = WORKDIR/self.base_name/fn
        return XArrayTarget(str(p))


class ExtractCumulantScaleProfiles(luigi.Task):
    base_names = luigi.Parameter()
    cumulants = luigi.Parameter()

    mask = luigi.Parameter(default=None)
    mask_args = luigi.Parameter(default='')

    def _parse_cumulant_arg(self):
        cums = [c.split(':') for c in self.cumulants.split(',')]
        return [c for (n,c) in enumerate(cums) if cums.index(c) == n]

    def requires(self):
        reqs = {}

        for base_name in self.base_names.split(','):
            reqs[base_name] = [
                ExtractCumulantScaleProfile(
                    base_name=base_name, v1=c[0], v2=c[1],
                    mask=self.mask, mask_args=self.mask_args,
                )
                for c in self._parse_cumulant_arg()
            ]

        return reqs

    def run(self):
        datasets = []
        for base_name in self.base_names.split(','):
            ds_ = xr.concat([
                input.open(decode_times=False)
                for input in self.input()[base_name]
            ], dim='cumulant')
            ds_['dataset_name'] = base_name
            datasets.append(ds_)

        ds = xr.concat(datasets, dim='dataset_name')
        ds.to_netcdf(self.output().fn)

    def output(self):
        unique_props = (self.base_names + self.cumulants)
        unique_identifier = hashlib.md5(unique_props.encode('utf-8')).hexdigest()
        fn = "cumulant_profile.{}.nc".format(unique_identifier)
        p = WORKDIR/fn
        return XArrayTarget(str(p))

class ExtractCrossSection2D(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    FN_FORMAT = "{exp_name}.out.xy.{field_name}.nc"


    def _extract_and_symlink_local_file(self):
        meta = _get_dataset_meta_info(self.base_name)

        p_out = Path(self.output().fn)
        p_in = Path(meta['path'])/"cross_sections"/"runtime_slices"/p_out.name

        assert p_in.exists()

        p_out.parent.mkdir(exist_ok=True, parents=True)

        os.symlink(str(p_in), str(p_out))

    def output(self):
        meta = _get_dataset_meta_info(self.base_name)

        fn = self.FN_FORMAT.format(
            exp_name=meta['experiment_name'],
            field_name=self.field_name
        )

        p = WORKDIR/self.base_name/"cross_sections"/"runtime_slices"/fn

        return luigi.LocalTarget(str(p))

    def run(self):
        meta = _get_dataset_meta_info(self.base_name)
        fn_out = self.output()

        if fn_out.exists():
            pass
        elif meta['host'] == 'localhost':
            self._extract_and_symlink_local_file()
        else:
            raise NotImplementedError(fn_out.fn)

class PerformObjectTracking2D(luigi.Task):
    base_name = luigi.Parameter()

    def requires(self):
        if not HAS_CLOUD_TRACKING:
            raise Exception("cloud_tracking_analysis module isn't available")

        return [
            ExtractCrossSection2D(base_name=self.base_name, field_name='core'),
            ExtractCrossSection2D(base_name=self.base_name, field_name='cldbase'),
            ExtractCrossSection2D(base_name=self.base_name, field_name='cldtop'),
            ExtractCrossSection2D(base_name=self.base_name, field_name='lwp'),
        ]

    def _get_tracking_identifier(self, meta):
        timestep_3d = meta['timestep']

        da_input = xr.open_dataarray(self.input()[0].fn, decode_times=False)
        # TODO: use more intelligent selection for timesteps to track here
        tn_max = len(da_input.time)

        return "track_1-{}".format(tn_max)

    def _get_dataset_name(self):
        meta = _get_dataset_meta_info(self.base_name)
        return meta['experiment_name']

    def get_cloud_data(self):
        meta = _get_dataset_meta_info(self.base_name)
        dataset_name = meta['experiment_name']

        tracking_identifier = self._get_tracking_identifier(meta)

        p_data = WORKDIR
        cloud_tracking_analysis.cloud_data.ROOT_DIR = str(p_data)
        cloud_data = CloudData(dataset_name, tracking_identifier,
                               dataset_pathname=self.base_name)

    def run(self):
        self.get_cloud_data()

    def output(self):
        if not all([i.exists() for i in self.input()]):
            return luigi.LocalTarget("fakefile.nc")

        meta = _get_dataset_meta_info(self.base_name)
        tracking_identifier = self._get_tracking_identifier(meta)
        tracking_identifier = tracking_identifier.replace('_', '__cloud_core__')

        dataset_name = meta['experiment_name']
        FN_2D_FORMAT = "{}.out.xy.{}.nc"

        fn = FN_2D_FORMAT.format(dataset_name, tracking_identifier)
        p = WORKDIR/self.base_name/"tracking_output"/fn
        return luigi.LocalTarget(str(p))


class ExtractCloudbaseState(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    cloud_age_max = luigi.FloatParameter(default=200.)

    def requires(self):
        if HAS_CLOUD_TRACKING:
            return dict(
                tracking=PerformObjectTracking2D(base_name=self.base_name),
                field=ExtractField3D(base_name=self.base_name, field_name=self.field_name),
            )
        else:
            warnings.warn("cloud tracking isn't available. Using approximate"
                          " method for finding cloud-base height rather than"
                          "tracking")
            return dict(
                qc=ExtractField3D(base_name=self.base_name, field_name='qc'),
                field=ExtractField3D(base_name=self.base_name, field_name=self.field_name),
            )

    def run(self):
        if HAS_CLOUD_TRACKING:
            tracking_output = Path(self.input()["tracking"].fn)
            matches = re.match("(.*).out.xy.track__cloud_core__(.*).nc", tracking_output.name)
            dataset_name = matches[1]
            tracking_timerange = matches[2]
            tracking_identifier = "track_{}".format(tracking_timerange)

            p_data = WORKDIR
            cloud_tracking_analysis.cloud_data.ROOT_DIR = str(p_data)
            cloud_data = CloudData(dataset_name, tracking_identifier,
                                   dataset_pathname=self.base_name)

            da_scalar_3d = xr.open_dataarray(self.input()["field"].fn, decode_times=False)

            t0 = da_scalar_3d.time.values[0]
            z_cb = cross_correlation_with_height.get_cloudbase_height(
                cloud_data=cloud_data, t0=t0, t_age_max=self.cloud_age_max,
            )
            dz = cloud_data.dx
            method = 'tracked clouds'
        else:
            qc = self.input()['qc'].open()
            z_cb = cross_correlation_with_height.get_approximate_cloudbase_height(
                qc=qc, z_tol=50.
            )
            da_scalar_3d = self.input()['field'].open()
            try:
                dz = find_vertical_grid_spacing(da_scalar_3d)
                method = 'approximate'
            except:
                warnings.warn("Using cloud-base state because vertical grid"
                              " spacing is non-uniform")
                dz = 0.0
                method = 'approximate, in-cloud'

        da_cb = cross_correlation_with_height.extract_from_3d_at_heights_in_2d(
            da_3d=da_scalar_3d, z_2d=z_cb-dz
        )
        da_cb = da_cb.squeeze()
        da_cb.name = self.field_name
        da_cb.attrs['method'] = method
        da_cb.attrs['cloud_age_max'] = self.cloud_age_max
        da_cb.attrs['num_clouds'] = z_cb.num_clouds

        da_cb.to_netcdf(self.output().fn)

    def output(self):
        fn = "{}.{}.max_t_age__{:.0f}s.cloudbase.xy.nc".format(self.base_name,
                                            self.field_name,
                                            self.cloud_age_max,
                                            )
        p = WORKDIR/self.base_name/fn
        return XArrayTarget(str(p))


class EstimateCharacteristicScales(luigi.Task):
    object_splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')
    variables = ['length', 'width', 'thickness']
    object_filters = luigi.Parameter(default=None)
    fit_type = luigi.Parameter(default='exponential')

    def requires(self):
        return ComputeObjectScales(
            variables=",".join(self.variables), base_name=self.base_name,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar,
            object_filters=self.object_filters,
        )

    def run(self):
        ds = self.input().open()
        assert self.fit_type == 'exponential'
        fn = length_scales.model_fitting.exponential_fit.fit
        ds_scales = ds[self.variables].apply(fit)
        ds_scales.to_netcdf(self.output().fn)

    def output(self):
        mask_name = MakeMask.make_mask_name(
            base_name=self.base_name,
            method_name=self.mask_method,
            method_extra_args=self.mask_method_extra_args
        )
        fn = '{}.{}.exp_fit_scales.nc'.format(
            self.base_name, mask_name
        )
        p = WORKDIR/self.base_name/fn
        target = XArrayTarget(str(p))
        return target

class ComputePerObjectProfiles(luigi.Task):
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')
    object_splitting_scalar = luigi.Parameter()

    field_name = luigi.Parameter()
    op = luigi.Parameter()
    z_max = luigi.FloatParameter(default=None)

    def requires(self):
        return dict(
            field=ExtractField3D(
                base_name=self.base_name,
                field_name=self.field_name,
                ),
            objects=IdentifyObjects(
                base_name=self.base_name,
                mask_method=self.mask_method,
                mask_method_extra_args=self.mask_method_extra_args,
                splitting_scalar=self.object_splitting_scalar,
                )
            )

    def run(self):
        input = self.input()
        da = input['field'].open().squeeze()
        da_objects = input['objects'].open()

        object_ids = np.unique(da_objects.chunk(None).values)
        if object_ids[0] == 0:
            object_ids = object_ids[1:]

        kwargs = dict(scalar=da.name, objects=da_objects.name,
                      object_ids=object_ids, op=self.op)

        ds = xr.merge([da, da_objects])

        if self.z_max is not None:
            ds = ds.sel(zt=slice(None, self.z_max))

        da_by_height = ds.groupby('zt').apply(self._apply_on_slice, kwargs=kwargs)

        da_by_height.to_netcdf(self.output().fn)

    @staticmethod
    def _apply_on_slice(ds, kwargs):
        da_ = ds[kwargs['scalar']].compute()
        da_objects_ = ds[kwargs['objects']].compute()
        object_ids = kwargs['object_ids']
        fn = getattr(dask_image.ndmeasure, kwargs['op'])
        v = fn(da_, labels=da_objects_, index=object_ids).compute()
        da = xr.DataArray(data=v, dims=['object_id',],
                            coords=dict(object_id=object_ids))
        da.name = "{}__{}".format(da_.name, kwargs['op'])
        da.attrs['units'] = da_.units
        da.attrs['long_name'] = '{} of {} per object'.format(
            kwargs['op'], da_.long_name,
        )
        return da

    def output(self):
        mask_name = MakeMask.make_mask_name(
            base_name=self.base_name,
            method_name=self.mask_method,
            method_extra_args=self.mask_method_extra_args
        )
        fn = "{base_name}.{mask_name}.{field_name}__{op}.by_z_per_object{ex}.nc".format(
            base_name=self.base_name, mask_name=mask_name,
            field_name=self.field_name, op=self.op,
            ex=self.z_max is None and "" or "_to_z" + str(self.z_max)
        )
        p = WORKDIR/self.base_name/fn
        target = XArrayTarget(str(p))
        return target

class ComputeObjectScaleVsHeightComposition(luigi.Task):
    x = luigi.Parameter()
    field_name = luigi.Parameter()

    base_name = luigi.Parameter()
    mask_method = luigi.Parameter()
    mask_method_extra_args = luigi.Parameter(default='')
    object_splitting_scalar = luigi.Parameter()

    object_filters = luigi.Parameter(default=None)
    z_max = luigi.FloatParameter(default=None)

    def requires(self):
        if "+" in self.base_name:
            # "+" can be used a shorthand to concat objects together from
            # different datasets
            base_names = self.base_name.split("+")
            reqs = dict([
                (base_name, ComputeObjectScaleVsHeightComposition(
                                base_name=base_name,
                                mask_method=self.mask_method,
                                mask_method_extra_args=self.mask_method_extra_args,
                                object_splitting_scalar=self.object_splitting_scalar,
                                object_filters=self.object_filters,
                                x=self.x,
                                field_name=self.field_name,
                                z_max=self.z_max,
                            ))
                for base_name in base_names
            ])
            return reqs
        else:
            return dict(
                decomp_profile=ComputePerObjectProfiles(
                    base_name=self.base_name,
                    mask_method=self.mask_method,
                    mask_method_extra_args=self.mask_method_extra_args,
                    object_splitting_scalar=self.object_splitting_scalar,
                    field_name=self.field_name,
                    op='sum',
                    z_max=self.z_max,
                ),
                da_3d=ExtractField3D(
                    base_name=self.base_name,
                    field_name=self.field_name,
                ),
                scales=ComputeObjectScales(
                    base_name=self.base_name,
                    mask_method=self.mask_method,
                    mask_method_extra_args=self.mask_method_extra_args,
                    object_splitting_scalar=self.object_splitting_scalar,
                    variables=self.x,
                    object_filters=self.object_filters,
                ),
                mask=MakeMask(
                    base_name=self.base_name,
                    method_name=self.mask_method,
                    method_extra_args=self.mask_method_extra_args,
                ),
            )

    def run(self):
        if "+" in self.base_name:
            dss = [input.open() for input in self.input().values()]
            if len(set([ds.nx for ds in dss])) != 1:
                raise Exception("All selected base_names must have same number"
                                " of points in x-direction (nx)")
            if len(set([ds.ny for ds in dss])) != 1:
                raise Exception("All selected base_names must have same number"
                                " of points in y-direction (ny)")
            # we increase the effective area so that the mean flux contribution
            # is scaled correctly, the idea is that we're just considering a
            # larger domain now and the inputs are stacked in the x-direction
            nx = np.sum([ds.nx for ds in dss])
            ny = dss[0].ny

            # strip out the reference profiles, these aren't concatenated but
            # instead we take the mean, we squeeze here so that for example
            # `time` isn't kept as a dimension
            das_profile = [ds.prof_ref.squeeze() for ds in dss]
            dss = [ds.drop('prof_ref') for ds in dss]

            # we use the mean profile across datasets for now
            da_prof_ref = xr.concat(das_profile, dim='base_name').mean(dim='base_name', dtype=np.float64)

            ds = merge_object_datasets(dss)
        else:
            input = self.input()
            da_field = input['decomp_profile'].open()
            da_3d = input['da_3d'].open()
            ds_scales = input['scales'].open()
            da_mask = input['mask'].open()
            nx, ny = da_3d.xt.count(), da_3d.yt.count()

            da_prof_ref = da_3d.where(da_mask).sum(dim=('xt', 'yt'),
                                                   dtype=np.float64)/(nx*ny)
            da_prof_ref.name = 'prof_ref'

            if self.object_filters is not None:
                da_field.where(da_field.object_id == ds_scales.object_id)

            ds = xr.merge([da_field, ds_scales])

            if self.z_max is not None:
                ds = ds.sel(zt=slice(None, self.z_max))

            ds = ds.where(np.logical_and(
                ~np.isinf(ds[self.x]),
                ~np.isnan(ds[self.x]),
            ), drop=True)

        ds_combined = xr.merge([ds, da_prof_ref])
        ds_combined.attrs['nx'] = int(nx)
        ds_combined.attrs['ny'] = int(ny)

        ds_combined = ds_combined.sel(zt=slice(None, self.z_max))

        fn = self.output().fn
        Path(fn).parent.mkdir(parents=True, exist_ok=True)
        ds_combined.to_netcdf(fn)

    def output(self):
        mask_name = MakeMask.make_mask_name(
            base_name=self.base_name,
            method_name=self.mask_method,
            method_extra_args=self.mask_method_extra_args
        )
        s_filter = ''
        if self.object_filters is not None:
            s_filter = '.filtered_by.{}'.format(
                (self.object_filters.replace(',','.')
                                    .replace(':', '__')
                                    .replace('=', '_')
                )
            )
        fn = ("{base_name}.{mask_name}.{field_name}__by__{x}"
             "{s_filter}.{filetype}".format(
            base_name=self.base_name, mask_name=mask_name,
            field_name=self.field_name, x=self.x, filetype="nc",
            s_filter=s_filter,
        ))

        p = WORKDIR/self.base_name/fn
        target = XArrayTarget(str(p))
        return target
