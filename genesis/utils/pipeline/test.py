import os
import subprocess

import luigi
import xarray as xr
import yaml

from .. import mask_functions, make_mask
from ... import objects


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
        try:
            with open('datasources.yaml') as fh:
                datasources = yaml.load(fh)
        except IOError:
            raise Exception("please define your data sources in datasources.yaml")

        if not self.base_name in datasources:
            raise Exception("Please make a definition for `{}` in "
                            "datasources.yaml".format(self.base_name))

        meta = datasources[self.base_name]

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
