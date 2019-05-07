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

    if not base_name in datasources:
        raise Exception("Please make a definition for `{}` in "
                        "datasources.yaml".format(base_name))

    return datasources[base_name]


class ExtractField3D(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    def run(self):
        fn_out = self.output()

        if fn_out.exists():
            pass
        else:
            raise NotImplementedError(fn_out.fn)

    def output(self):
        meta = _get_dataset_meta_info(self.base_name)

        fn_out = "{exp_name}.tn{timestep}.{field_name}.nc".format(
            exp_name=meta['experiment_name'], timestep=meta['timestep'],
            field_name=self.field_name
        )

        return luigi.LocalTarget(fn_out)


class MakeRadTracerMask(luigi.Task):
    cutoff = luigi.FloatParameter()
    base_name = luigi.Parameter()
    method_name = 'rad_tracer_thermals'

    def requires(self):
        return ExtractField3D(field_name='cvrxp', base_name=self.base_name)

    def run(self):
        mask_fn = getattr(mask_functions, self.method_name)
        mask = mask_fn(
            base_name=self.base_name, cvrxp=xr.open_dataarray(self.input().fn),
            num_std_div=self.cutoff
        )
        mask.to_netcdf(self.output().fn)

    def output(self):
        return luigi.LocalTarget(make_mask.OUT_FILENAME_FORMAT.format(
            base_name=self.base_name, method_name=self.method_name
        ))


class IdentifyObjects(luigi.Task):
    splitting_scalar = luigi.Parameter()
    mask_cutoff = luigi.FloatParameter()
    base_name = luigi.Parameter()

    def requires(self):
        return dict(
            mask=MakeRadTracerMask(
                base_name=self.base_name,
                cutoff=self.mask_cutoff,
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
        mask_name = make_mask.OUT_FILENAME_FORMAT.format(
            base_name=self.base_name, method_name=MakeRadTracerMask.method_name
        )

        return luigi.LocalTarget(objects.identify.OUT_FILENAME_FORMAT.format(
            base_name=self.base_name, mask_name=mask_name
        ))

class ComputeObjectScale(luigi.Task):
    object_splitting_scalar = luigi.Parameter()
    mask_cutoff = luigi.Parameter()
    base_name = luigi.Parameter()

    variable = luigi.Parameter()
    operator = luigi.Parameter(default=None)

    def requires(self):
        return IdentifyObjects(
            base_name=self.base_name,
            mask_cutoff=self.mask_cutoff,
            splitting_scalar=self.object_splitting_scalar,
        )

    def output(self):
        mask_name = make_mask.OUT_FILENAME_FORMAT.format(
            base_name=self.base_name, method_name=MakeRadTracerMask.method_name
        )

        fn = objects.integrate.make_output_filename(
            base_name=self.base_name, mask_identifier=mask_name,
            variable=self.variable, operator=self.operator
        )
        return luigi.LocalTarget(fn)

    def run(self):
        da_objects = xr.open_dataarray(self.input().fn)

        ds = objects.integrate.integrate(objects=da_objects,
                                         variable=self.variable)
        ds.to_netcdf(self.output().fn)


class ComputeObjectScales(luigi.Task):
    object_splitting_scalar = luigi.Parameter()
    mask_cutoff = luigi.FloatParameter()
    base_name = luigi.Parameter()

    variables = [
        ('com_angles', None),
        ('volume', None),
    ]

    def requires(self):
        return [
            ComputeObjectScale(
                base_name=self.base_name,
                mask_cutoff=self.mask_cutoff,
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
