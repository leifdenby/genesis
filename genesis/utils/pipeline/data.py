import os
import subprocess
from pathlib import Path

import luigi
import xarray as xr
import yaml

from .. import mask_functions, make_mask
from ... import objects
from ...bulk_statistics import cross_correlation_with_height

import cloud_tracking_analysis.cloud_data
from cloud_tracking_analysis import CloudData, CloudType, cloud_operations

def _get_dataset_meta_info(base_name):
    try:
        with open('datasources.yaml') as fh:
            datasources = yaml.load(fh)
    except IOError:
        raise Exception("please define your data sources in datasources.yaml")

    if datasources is None or not base_name in datasources:
        raise Exception("Please make a definition for `{}` in "
                        "datasources.yaml".format(base_name))

    return datasources[base_name]


class ExtractField3D(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    FN_FORMAT = "{exp_name}.tn{timestep}.{field_name}.nc"


    def _extract_and_symlink_local_file(self):
        meta = _get_dataset_meta_info(self.base_name)

        p_out = Path(self.output().fn)
        p_in = Path(meta['path'])/"3d_blocks"/"full_domain"/p_out.name

        p_out.parent.mkdir(exist_ok=True, parents=True)

        os.symlink(str(p_in), str(p_out))

    def run(self):
        meta = _get_dataset_meta_info(self.base_name)

        fn_out = self.output()

        if fn_out.exists():
            pass
        elif meta['host'] == 'localhost':
            self._extract_and_symlink_local_file()
        else:
            raise NotImplementedError(fn_out.fn)

    def output(self):
        meta = _get_dataset_meta_info(self.base_name)

        fn = self.FN_FORMAT.format(
            exp_name=meta['experiment_name'], timestep=meta['timestep'],
            field_name=self.field_name
        )

        p = Path("data")/self.base_name/fn

        return luigi.LocalTarget(str(p))


class MakeMask(luigi.Task):
    base_name = luigi.Parameter()
    method_extra_args = luigi.Parameter(default='')
    method_name = luigi.Parameter()

    def requires(self):
        method_kwargs = self._build_method_kwargs()
        try:
            make_mask.build_method_kwargs(method=self.method_name, kwargs=method_kwargs)
        except make_mask.MissingInputException as e:
            return dict([
                (v, ExtractField3D(field_name=v, base_name=self.base_name))
                for v in e.missing_kwargs
            ])

    def _build_method_kwargs(self):
        kwargs = dict(base_name=self.base_name)
        for kv in self.method_extra_args.split(","):
            k,v = kv.split("=")
            kwargs[k] = v
        return kwargs

    def run(self):
        method_kwargs = self._build_method_kwargs()

        for (v, target) in self.input().items():
            method_kwargs[v] = xr.open_dataarray(target.fn, decode_times=False)

        mask = make_mask.main(method=self.method_name, method_kwargs=method_kwargs)
        mask.to_netcdf(self.output().fn)

    def output(self):
        kwargs = self._build_method_kwargs()

        try:
            kwargs = make_mask.build_method_kwargs(
                method=self.method_name, kwargs=kwargs
            )
        except make_mask.MissingInputException as e:
            for v in e.missing_kwargs:
                kwargs[v] = None
        kwargs = make_mask.build_method_kwargs(
            method=self.method_name, kwargs=kwargs
        )

        mask_name = make_mask.mask_mask_name(
            method=self.method_name, method_kwargs=kwargs
        )
        return luigi.LocalTarget(make_mask.OUT_FILENAME_FORMAT.format(
            base_name=self.base_name, mask_name=mask_name
        ))


class IdentifyObjects(luigi.Task):
    splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()
    mask_method = luigi.Parameter(default='rad_tracer_thermals')
    mask_method_extra_args = luigi.Parameter(default='num_std_div=3.0')

    def requires(self):
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
        da_mask = xr.open_dataarray(self.input()['mask'].fn).squeeze()
        da_scalar = xr.open_dataarray(self.input()['scalar'].fn).squeeze()

        object_labels = objects.identify.process(
            mask=da_mask, splitting_scalar=da_scalar
        )

        object_labels.to_netcdf(self.output().fn)

    def output(self):
        if not self.input()["mask"].exists():
            return luigi.LocalTarget("fakefile.nc")

        da_mask = xr.open_dataarray(self.input()["mask"].fn, decode_times=False)
        mask_name = da_mask.name
        objects_name = objects.identify.make_objects_name(
            mask_name=mask_name, splitting_var=self.splitting_scalar
        )

        return luigi.LocalTarget(objects.identify.OUT_FILENAME_FORMAT.format(
            base_name=self.base_name, objects_name=objects_name
        ))

class ComputeObjectScale(luigi.Task):
    object_splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()

    variable = luigi.Parameter()
    operator = luigi.Parameter(default=None)

    def requires(self):
        return IdentifyObjects(
            base_name=self.base_name,
            splitting_scalar=self.object_splitting_scalar,
        )

    def output(self):
        if not self.input().exists():
            return luigi.LocalTarget("fakefile.nc")

        da_objects = xr.open_dataarray(self.input().fn, decode_times=False)
        objects_name = da_objects.name

        name = objects.integrate.make_name(variable=self.variable,
                                           operator=self.operator)

        fn = objects.integrate.FN_OUT_FORMAT.format(
            base_name=self.base_name, objects_name=objects_name,
            name=name
        )
        return luigi.LocalTarget(fn)

    def run(self):
        da_objects = xr.open_dataarray(self.input().fn)

        ds = objects.integrate.integrate(objects=da_objects,
                                         variable=self.variable)
        ds.to_netcdf(self.output().fn)


class ComputeObjectScales(luigi.Task):
    object_splitting_scalar = luigi.Parameter()
    base_name = luigi.Parameter()

    variables = [
        ('com_angles', None),
        ('volume', None),
    ]

    def requires(self):
        return [
            ComputeObjectScale(
                base_name=self.base_name,
                object_splitting_scalar=self.object_splitting_scalar,
                variable=v, operator=op) 
            for (v, op) in self.variables
        ]

    def run(self):
        pass

class ComputeCumulantProfiles(luigi.Task):
    pass


class ExtractCrossSection2D(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    def output(self):
        meta = _get_dataset_meta_info(self.base_name)
        dataset_name = meta['experiment_name']
        FN_2D_FORMAT = "{}.out.xy.{}.nc"

        fn = FN_2D_FORMAT.format(dataset_name, self.field_name)
        p = Path(os.getcwd())/self.base_name/"cross_sections"/"runtime_slices"/fn

        return luigi.LocalTarget(str(p))

    def run(self):
        if not self.output().exists():
            raise Exception("Please copy data for `{}` to `{}`".format(
                self.field_name, self.output().fn
            ))

class PerformObjectTracking2D(luigi.Task):
    base_name = luigi.Parameter()

    def requires(self):
        return [
            ExtractCrossSection2D(base_name=self.base_name, field_name='core'),
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

        cloud_tracking_analysis.cloud_data.ROOT_DIR = os.getcwd()
        cloud_data = CloudData(dataset_name, tracking_identifier,
                               dataset_pathname=self.base_name)

    def run(self):
        self.get_cloud_data()

    def output(self):
        meta = _get_dataset_meta_info(self.base_name)
        tracking_identifier = self._get_tracking_identifier(meta)

        dataset_name = meta['experiment_name']
        FN_2D_FORMAT = "{}.out.xy.{}.nc"

        fn = FN_2D_FORMAT.format(dataset_name, tracking_identifier)
        p = Path(os.getcwd())/self.base_name/"tracking_output"/fn
        return luigi.LocalTarget(str(p))


class ExtractCloudbaseState(luigi.Task):
    base_name = luigi.Parameter()

    def requires(self):
        return [
            PerformObjectTracking2D(base_name=self.base_name),
            ExtractField3D(
                base_name=self.base_name,
                field_name="cvrxp"  # actually only needed for the timestep...
            )
        ]

    def run(self):
        tracking_output = Path(self.input()[0].fn)
        dataset_name = tracking_output.name.split('.')[0]
        tracking_identifier = tracking_output.name.split('.')[-2]

        cloud_tracking_analysis.cloud_data.ROOT_DIR = os.getcwd()
        cloud_data = CloudData(dataset_name, tracking_identifier,
                               dataset_pathname=self.base_name)

        da_scalar_3d = xr.open_dataarray(self.input()[1].fn, decode_times=False)

        t0 = da_scalar_3d.time.values[0]
        import ipdb
        with ipdb.launch_ipdb_on_exception():
            ds_cb = cross_correlation_with_height.get_cloudbase_data(cloud_data=cloud_data, t0=t0)

        ds_cb.to_netcdf(self.output().fn)


    def output(self):
        fn = "{}.cloudbase.xy.nc".format(self.base_name)
