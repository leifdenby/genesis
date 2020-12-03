# coding: utf-8
import numpy as np
import xarray as xr
import luigi
from pathlib import Path

from ....utils.transforms import offset_gal
from . import ExtractField3D
from .base import (
    NumpyDatetimeParameter,
    XArrayTarget,
    _get_dataset_meta_info,
    get_workdir,
)


class Single3DShiftedCrop(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()
    t_ref = NumpyDatetimeParameter()
    center_pt = luigi.ListParameter()
    l_win = luigi.FloatParameter()

    def requires(self):
        assert len(self.center_pt) == 2
        return ExtractField3D(
            field_name=self.field_name,
            base_name=self.base_name,
        )

    def run(self):
        base_name = self.base_name.split(".tn")[0]
        meta = _get_dataset_meta_info(base_name=base_name)
        U_gal = meta["U_gal"]
        da = self.input().open()
        da_shifted = offset_gal(
            da=da, U=U_gal, tref=np.datetime64(self.t_ref), truncate_to_grid=True
        )
        l_win = self.l_win
        x_c, y_c = self.center_pt
        crop_kwargs = dict(
            xt=slice(x_c - l_win / 2.0, x_c + l_win / 2.0),
            yt=slice(y_c - l_win / 2.0, y_c + l_win / 2.0),
        )
        da_crop = da_shifted.sel(**crop_kwargs)
        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_crop.to_netcdf(self.output().fn)

    def output(self):
        name_parts = [
            self.base_name,
            self.field_name,
            f"crop__{self.center_pt[0]}_{self.center_pt[1]}_{self.l_win}",
            "nc",
        ]
        fn = ".".join(name_parts)
        p = get_workdir() / self.base_name / "3d_crops" / fn
        return XArrayTarget(str(p))


class Timespan3DShiftedCrop(luigi.Task):
    base_name = luigi.Parameter()
    timesteps = luigi.ListParameter()
    field_name = luigi.Parameter(default="qc")
    t_ref = NumpyDatetimeParameter()
    center_pt = luigi.ListParameter()
    l_win = luigi.FloatParameter(default=2.0e3)

    def requires(self):
        tasks = []
        for tn in self.timesteps:
            bs = f"{self.base_name}.tn{tn}"
            t = Single3DShiftedCrop(
                base_name=bs,
                t_ref=self.t_ref,
                l_win=self.l_win,
                center_pt=self.center_pt,
                field_name=self.field_name,
            )
            tasks.append(t)
        return tasks

    def run(self):
        das = [o.open() for o in self.input()]
        da_combined = xr.concat(das, dim="time")

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_combined.to_netcdf(self.output().fn)

    def output(self):
        name_parts = [
            self.base_name,
            "_".join([str(v) for v in self.timesteps]),
            self.field_name,
            f"crop__{self.center_pt[0]}_{self.center_pt[1]}_{self.l_win}",
            "nc",
        ]
        fn = ".".join(name_parts)
        p = get_workdir() / self.base_name / "3d_crops" / fn
        return XArrayTarget(str(p))


class Timespan3DShiftedCropCollection(luigi.Task):
    base_name = luigi.Parameter()
    crop_base_name_suffix = luigi.Parameter(default=None)
    tn_min = luigi.IntParameter()
    tn_max = luigi.IntParameter()
    field_names = luigi.ListParameter(
        default=[
            "qc",
            "w",
            "theta_l",
            "qv__norain",
            "d_theta_l",
            "d_qv__norain",
            "cvrxp_p_stddivs",
        ]
    )
    t_ref = NumpyDatetimeParameter()
    center_pt = luigi.ListParameter()
    l_win = luigi.FloatParameter(default=2.0e3)
    store_in_rundir = luigi.BoolParameter(default=False)
    z_max = luigi.Parameter(default=None)

    def requires(self):
        t_ref = self.t_ref
        timesteps = np.arange(self.tn_min, self.tn_max)

        tasks = []
        for field_name in self.field_names:
            t = Timespan3DShiftedCrop(
                base_name=self.base_name,
                t_ref=t_ref,
                timesteps=timesteps.tolist(),
                field_name=field_name,
                center_pt=self.center_pt,
                l_win=self.l_win,
            )
            tasks.append(t)
        return tasks

    def run(self):
        das = [o.open() for o in self.input()]
        ds_combined = xr.merge(das)

        if self.z_max is not None:
            ds_combined = ds_combined.sel(zt=slice(None, self.z_max))

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        ds_combined.to_netcdf(self.output().fn)

    def output(self):
        name_parts = [
            self.base_name,
            f"{self.tn_min}__{self.tn_max}",
            "_".join(self.field_names),
            f"crop__{self.center_pt[0]}_{self.center_pt[1]}_{self.l_win}",
            "nc",
        ]

        if self.z_max is not None:
            name_parts.insert(-2, f"z_max__{self.z_max}")

        fn = ".".join(name_parts)
        if self.store_in_rundir:
            p = Path(".") / fn
        else:
            p = get_workdir() / self.base_name / "3d_crops" / fn
        return XArrayTarget(str(p))
