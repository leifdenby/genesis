import os

import luigi
import xarray as xr

from ... import make_mask
from .extraction import ExtractField3D
from .base import WORKDIR, XArrayTarget


class MakeMask(luigi.Task):
    base_name = luigi.Parameter()
    method_extra_args = luigi.Parameter(default='')
    method_name = luigi.Parameter()

    def requires(self):
        reqs = {}
        if "__filtered_by=" in self.method_name:
            raise Exception("Use `MakeMaskWithObjects` instead")

        method_name = self.method_name
        method_kwargs = self._build_method_kwargs(
            base_name=self.base_name, method_extra_args=self.method_extra_args
        )

        try:
            make_mask.build_method_kwargs(method=method_name,
                                          kwargs=method_kwargs)
        except make_mask.MissingInputException as e:
            for v in e.missing_kwargs:
                reqs[v] = ExtractField3D(field_name=v,
                                         base_name=self.base_name)

        return reqs

    @staticmethod
    def _build_method_kwargs(base_name, method_extra_args):
        kwargs = dict(base_name=base_name)
        for kv in method_extra_args.split(","):
            if kv == "":
                continue
            elif kv.startswith("object_splitting_scalar="):
                continue
            k, v = kv.split("=")
            kwargs[k] = v
        return kwargs

    def run(self):
        method_kwargs = self._build_method_kwargs(
            base_name=self.base_name, method_extra_args=self.method_extra_args
        )

        for (v, input) in self.input().items():
            method_kwargs[v] = xr.open_dataarray(input.fn, decode_times=False)

        cwd = os.getcwd()
        p_data = WORKDIR/self.base_name
        os.chdir(p_data)
        mask = make_mask.main(
            method=self.method_name, method_kwargs=method_kwargs
        )
        os.chdir(cwd)
        mask.to_netcdf(self.output().fn)

    @classmethod
    def make_mask_name(cls, base_name, method_name, method_extra_args):
        kwargs = cls._build_method_kwargs(
            base_name=base_name, method_extra_args=method_extra_args
        )

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
