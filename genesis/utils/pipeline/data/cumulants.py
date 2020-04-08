import hashlib

import xarray as xr
import luigi

from ... import length_scales
from .extraction import ExtractField3D
from .masking import MakeMask
from .base import WORKDIR, XArrayTarget


class ExtractCumulantScaleProfile(luigi.Task):
    base_name = luigi.Parameter()
    v1 = luigi.Parameter()
    v2 = luigi.Parameter()
    z_max = luigi.FloatParameter(default=700.)
    mask = luigi.Parameter(default=None)
    mask_args = luigi.Parameter(default='')
    width_method = (length_scales.cumulant.calc
                                 .WidthEstimationMethod.MASS_WEIGHTED)

    def requires(self):
        reqs = {}
        reqs['fields'] = [
                ExtractField3D(
                    base_name=self.base_name,
                    field_name=self.v1
                ),
                ExtractField3D(
                    base_name=self.base_name,
                    field_name=self.v2
                ),
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

        calc_fn = (
            length_scales
            .cumulant
            .vertical_profile
            .calc
            .get_height_variation_of_characteristic_scales
        )

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
        return [c for (n, c) in enumerate(cums) if cums.index(c) == n]

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
        unique_identifier = hashlib.md5(
            unique_props.encode('utf-8')
        ).hexdigest()
        fn = "cumulant_profile.{}.nc".format(unique_identifier)
        p = WORKDIR/fn
        return XArrayTarget(str(p))
