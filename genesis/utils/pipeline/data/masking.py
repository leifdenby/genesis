import os
import warnings
from pathlib import Path

import luigi
import xarray as xr

from ... import make_mask
from .extraction import ExtractField3D, ExtractCrossSection2D
from .base import get_workdir, XArrayTarget
from .base import _get_dataset_meta_info, XArrayTargetUCLALES


class Find3DTimesteps(luigi.Task):
    base_name = luigi.Parameter()

    def run(self):
        meta = _get_dataset_meta_info(self.base_name)
        p_in = Path(meta["path"]) / "raw_data" / f"{self.base_name}.00000000.nc"
        da_3d_block = XArrayTargetUCLALES(str(p_in)).open()
        da_3d_block.time.to_netcdf(self.output().fn)

    def output(self):
        fn = f"{self.base_name}.timesteps.nc"
        p = get_workdir() / self.base_name / fn

        return XArrayTarget(str(p))


class ExtractCrossSection2DFrom3DTimestep(luigi.Task):
    """
    Extract the 2D cross-section for timestep `tn` in
    base_name=`{base_name}.tn{tn}`
    """

    var_name = luigi.Parameter()
    base_name = luigi.Parameter()

    def _parse_basename(self):
        assert ".tn" in self.base_name
        bn, tn = self.base_name.split(".tn")
        return bn, int(tn)

    def requires(self):
        bn, tn = self._parse_basename()
        return Find3DTimesteps(base_name=bn)

    def run(self):
        da_3d_timesteps = self.input().open()
        bn, tn = self._parse_basename()

        da_time = da_3d_timesteps.isel(time=tn)
        output_2d = yield ExtractCrossSection2D(
            var_name=self.var_name, base_name=bn, time=da_time
        )
        da_2d = output_2d.open()
        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_2d.to_netcdf(self.output().fn)

    def output(self):
        fn = f"{self.base_name}_3d.{self.var_name}.nc"
        p = get_workdir() / self.base_name / "cross_sections" / "runtime_slices" / fn
        return XArrayTarget(str(p))


class MakeMask(luigi.Task):
    base_name = luigi.Parameter()
    method_extra_args = luigi.Parameter(default="")
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
            make_mask.build_method_kwargs(method=method_name, kwargs=method_kwargs)
        except make_mask.MissingInputException as e:
            for v in e.missing_kwargs:
                if v.endswith("__2d"):
                    v_2d = v[:-4]
                    reqs[v] = ExtractCrossSection2DFrom3DTimestep(
                        var_name=v_2d, base_name=self.base_name
                    )
                else:
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
        p_data = get_workdir() / self.base_name
        os.chdir(p_data)
        mask = make_mask.main(method=self.method_name, method_kwargs=method_kwargs)
        os.chdir(cwd)
        mask.to_netcdf(self.output().fn)

    @classmethod
    def make_mask_name(cls, base_name, method_name, method_extra_args):
        kwargs = cls._build_method_kwargs(
            base_name=base_name, method_extra_args=method_extra_args
        )

        try:
            kwargs = make_mask.build_method_kwargs(method=method_name, kwargs=kwargs)
        except make_mask.MissingInputException as e:
            for v in e.missing_kwargs:
                kwargs[v] = None
        kwargs = make_mask.build_method_kwargs(method=method_name, kwargs=kwargs)

        mask_name = make_mask.make_mask_name(method=method_name, method_kwargs=kwargs)

        return mask_name

    def output(self):
        mask_name = self.make_mask_name(
            base_name=self.base_name,
            method_name=self.method_name,
            method_extra_args=self.method_extra_args,
        )

        fn = make_mask.OUT_FILENAME_FORMAT.format(
            base_name=self.base_name, mask_name=mask_name
        )
        p = get_workdir() / self.base_name / fn
        return XArrayTarget(str(p))
