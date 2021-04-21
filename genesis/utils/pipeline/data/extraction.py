import os
from pathlib import Path
import warnings
import importlib
import re

import luigi
import xarray as xr
import numpy as np
from tqdm import tqdm

from ....utils import calc_flux, transforms, find_vertical_grid_spacing
from ... import mask_functions
from .base import (
    get_workdir,
    XArrayTarget,
    _get_dataset_meta_info,
    NumpyDatetimeParameter,
)


if "USE_SCHEDULER" in os.environ:
    from dask.distributed import Client

    client = Client(threads_per_worker=1)

REGEX_INSTANTENOUS_BASENAME = re.compile(r"(?P<base_name_2d>.*)\.tn(?P<timestep>\d+)")

COMPOSITE_FIELD_METHODS = dict(
    p_stddivs=(mask_functions.calc_scalar_perturbation_in_std_div, []),
    flux=(calc_flux.compute_vertical_flux, ["w"]),
    _prefix__d=(calc_flux.get_horz_devition, []),
)


def fix_time_units(da):
    modified = False
    if np.issubdtype(da.dtype, np.datetime64):
        # already converted since xarray has managed to parse the time in
        # CF-format
        pass
    elif da.attrs["units"].startswith("seconds since 2000-01-01"):
        # I fixed UCLALES to CF valid output, this is output from a fixed
        # version
        pass
    elif da.attrs["units"].startswith("seconds since 2000-00-00"):
        da.attrs["units"] = da.attrs["units"].replace(
            "seconds since 2000-00-00",
            "seconds since 2000-01-01",
        )
        modified = True
    elif da.attrs["units"].startswith("seconds since 0-00-00"):
        # 2D fields have strange time units...
        da.attrs["units"] = da.attrs["units"].replace(
            "seconds since 0-00-00",
            "seconds since 2000-01-01",
        )
        modified = True
    elif da.attrs["units"].startswith("seconds since 0-0-0"):
        # 2D fields have strange time units...
        da.attrs["units"] = da.attrs["units"].replace(
            "seconds since 0-0-0",
            "seconds since 2000-01-01",
        )
        modified = True
    elif da.attrs["units"] == "day as %Y%m%d.%f":
        da = (da * 24 * 60 * 60).astype(int)
        da.attrs["units"] = "seconds since 2000-01-01 00:00:00"
        modified = True
    else:
        raise NotImplementedError(da.attrs["units"])
    return da, modified


class XArrayTarget3DExtraction(XArrayTarget):
    def open(self, *args, **kwargs):
        ds = super(XArrayTarget3DExtraction, self).open(*args, **kwargs)
        if len(ds.coords) == 0:
            raise Exception(f"{self.fn} doesn't contain any data")
        ds = self._ensure_coord_units(ds)
        return ds

    def _ensure_coord_units(self, da):
        coord_names = ["xt", "yt"]
        for v in coord_names:
            if not "units" in da[v].attrs:
                warnings.warn(
                    f"The coordinate `{v}` for `{da.name}` is missing "
                    "units which are required for the cumulant calculation. "
                    "Assuming `meters`"
                )
                da_ = da[v]
                da_.attrs["units"] = "m"
                da = da.assign_coords(**{v: da_})
        return da


class ExtractField3D(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    FN_FORMAT = "{experiment_name}.{field_name}.nc"

    @staticmethod
    def _get_data_loader_module(meta):
        model_name = meta.get("model")
        if model_name is None:
            model_name = "UCLALES"

        module_name = ".data_sources.{}".format(model_name.lower().replace("-", "_"))
        return importlib.import_module(module_name, package="genesis.utils.pipeline")

    def requires(self):
        meta = _get_dataset_meta_info(self.base_name)
        data_loader = self._get_data_loader_module(meta=meta)

        reqs = {}

        derived_fields = getattr(data_loader, "DERIVED_FIELDS", None)

        if derived_fields is not None:
            for req_field in derived_fields.get(self.field_name, []):
                reqs[req_field] = ExtractField3D(
                    base_name=self.base_name, field_name=req_field
                )

        for (affix, (func, extra_fields)) in COMPOSITE_FIELD_METHODS.items():
            req_field = None
            if affix.startswith("_prefix__"):
                prefix = affix.replace("_prefix__", "")
                if self.field_name.startswith(prefix):
                    req_field = self.field_name.replace("{}_".format(prefix), "")
            else:
                postfix = affix
                if self.field_name.endswith(postfix):
                    req_field = self.field_name.replace("_{}".format(postfix), "")

            if req_field is not None:
                reqs["da"] = ExtractField3D(
                    base_name=self.base_name, field_name=req_field
                )
                for v in extra_fields:
                    reqs[v] = ExtractField3D(base_name=self.base_name, field_name=v)

        return reqs

    def run(self):
        meta = _get_dataset_meta_info(self.base_name)

        output = self.output()

        if output.exists():
            pass
        elif meta["host"] == "localhost":
            p_out = Path(self.output().fn)
            p_out.parent.mkdir(parents=True, exist_ok=True)

            is_composite = False
            for (affix, (func, _)) in COMPOSITE_FIELD_METHODS.items():
                if affix.startswith("_prefix__"):
                    prefix = affix.replace("_prefix__", "")
                    is_composite = self.field_name.startswith(prefix)
                else:
                    postfix = affix
                    is_composite = self.field_name.endswith(postfix)

                if is_composite:
                    das_input = dict(
                        [
                            (k, input.open(decode_times=False))
                            for (k, input) in self.input().items()
                        ]
                    )
                    da = func(**das_input)
                    # XXX: remove infs for now
                    da = da.where(~np.isinf(da))
                    da.to_netcdf(self.output().fn)
                    break

            if not is_composite:
                opened_inputs = dict(
                    [(k, input.open()) for (k, input) in self.input().items()]
                )
                data_loader = self._get_data_loader_module(meta=meta)
                result = data_loader.extract_field_to_filename(
                    dataset_meta=meta,
                    path_out=p_out,
                    field_name=self.field_name,
                    **opened_inputs,
                )
                if isinstance(result, luigi.Task):
                    yield result
                    data_loader.extract_field_to_filename(
                        dataset_meta=meta,
                        path_out=p_out,
                        field_name=self.field_name,
                        **opened_inputs,
                    )
        else:
            raise NotImplementedError(output.fn)

    def output(self):
        meta = _get_dataset_meta_info(self.base_name)

        fn = self.FN_FORMAT.format(
            experiment_name=meta["experiment_name"],
            timestep=meta["timestep"],
            field_name=self.field_name,
        )

        p = get_workdir() / self.base_name / fn

        t = XArrayTarget3DExtraction(str(p))

        if t.exists():
            data = t.open()
            if isinstance(data, xr.Dataset):
                if len(data.variables) == 0:
                    warnings.warn(
                        "Stored file for `{}` is empty, deleting..."
                        "".format(self.field_name)
                    )
                    p.unlink()

        return t


class XArrayTarget2DCrossSection(XArrayTarget):
    def open(self, *args, **kwargs):
        kwargs["decode_times"] = False
        da = super().open(*args, **kwargs)
        da["time"], _ = fix_time_units(da["time"])

        # xr.decode_cf only works on datasets
        ds = xr.decode_cf(da.to_dataset())
        da = ds[da.name]

        return da


class TimeCrossSectionSlices2D(luigi.Task):
    base_name = luigi.Parameter()
    var_name = luigi.Parameter()

    FN_FORMAT = "{exp_name}.out.xy.{var_name}.nc"

    def _extract_and_symlink_local_file(self):
        meta = _get_dataset_meta_info(self.base_name)

        p_out = Path(self.output().fn)
        p_in = Path(meta["path"]) / "cross_sections" / "runtime_slices" / p_out.name

        if not p_in.exists():
            raise FileNotFoundError(f"Couldn't find file {p_in} to symlink")

        p_out.parent.mkdir(exist_ok=True, parents=True)
        os.symlink(str(p_in.absolute()), str(p_out))

    def output(self):
        if REGEX_INSTANTENOUS_BASENAME.match(self.base_name):
            raise Exception(
                "Shouldn't pass base_name with timestep suffix"
                " (`.tn`) to tracking util"
            )

        meta = _get_dataset_meta_info(self.base_name)

        fn = self.FN_FORMAT.format(
            exp_name=meta["experiment_name"], var_name=self.var_name
        )

        p = get_workdir() / self.base_name / "cross_sections" / "runtime_slices" / fn

        return XArrayTarget2DCrossSection(str(p))

    def run(self):
        meta = _get_dataset_meta_info(self.base_name)
        fn_out = self.output()

        if fn_out.exists():
            pass
        elif meta["host"] == "localhost":
            self._extract_and_symlink_local_file()
        else:
            raise NotImplementedError(fn_out.fn)


class ProfileStatistics(luigi.Task):
    base_name = luigi.Parameter()

    FN_FORMAT = "{exp_name}.ps.nc"

    def _extract_and_symlink_local_file(self):
        meta = _get_dataset_meta_info(self.base_name)

        p_out = Path(self.output().fn)
        p_in = Path(meta["path"]) / "other" / p_out.name

        if not p_in.exists():
            raise FileNotFoundError(f"Couldn't find file {p_in} to symlink")

        p_out.parent.mkdir(exist_ok=True, parents=True)
        os.symlink(str(p_in.absolute()), str(p_out))

    def output(self):
        if REGEX_INSTANTENOUS_BASENAME.match(self.base_name):
            raise Exception(
                "Shouldn't pass base_name with timestep suffix"
                " (`.tn`) to get profile statistics"
            )

        meta = _get_dataset_meta_info(self.base_name)

        fn = self.FN_FORMAT.format(
            exp_name=meta["experiment_name"],
        )

        p = get_workdir() / self.base_name / fn

        return XArrayTarget(str(p))

    def run(self):
        meta = _get_dataset_meta_info(self.base_name)
        fn_out = self.output()

        if fn_out.exists():
            pass
        elif meta["host"] == "localhost":
            self._extract_and_symlink_local_file()
        else:
            raise NotImplementedError(fn_out.fn)


def remove_gal_transform(da, tref, base_name):
    meta = _get_dataset_meta_info(base_name)
    U_gal = meta.get("U_gal", None)
    if U_gal is None:
        raise Exception(
            "To remove the Galilean transformation"
            " please define the transform velocity"
            " as `U_gal` in datasources.yaml for"
            " dataset `{}`".format(base_name)
        )

    kws = dict(U=U_gal, tref=tref, truncate_to_grid=True)
    if da.time.count() > 1:
        return da.groupby("time").apply(transforms.offset_gal, **kws)
    else:
        return transforms.offset_gal(da=da, **kws)


class ExtractCrossSection2D(luigi.Task):
    base_name = luigi.Parameter()
    var_name = luigi.Parameter()
    time = NumpyDatetimeParameter()
    remove_gal_transform = luigi.BoolParameter(default=False)

    FN_FORMAT = "{exp_name}.out.xy.{var_name}.nc"

    def requires(self):
        return TimeCrossSectionSlices2D(
            base_name=self.base_name,
            var_name=self.var_name,
        )

    def _ensure_has_coord(self, da, coord):
        assert coord in ["xt", "yt"]
        if not coord in da.coords:
            meta = _get_dataset_meta_info(self.base_name)
            dx = meta.get("dx")
            if dx is None:
                raise Exception(
                    f"The grid variable `{coord}` is missing from the"
                    f" `{self.var_name}` 2D cross-section field. To"
                    " create the missing grid coordinate you need"
                    " to define the variable `dx` in the dataset"
                    " meta information"
                )

            nx = int(da[coord].count())
            x = dx * np.arange(-nx // 2, nx // 2)
            da.coords[coord] = (coord,), x
            warnings.warn(
                f"Coordinate values for `{coord}` is missing for variable"
                f" `{self.var_name}`, creating it from `dx` in meta-info"
                f" for `{self.base_name}`"
            )
            da.coords[coord].attrs["units"] = "m"
            da.coords[coord].attrs["long_name"] = "cell-center position"

    def run(self):
        da_timedep = self.input().open()
        da = da_timedep.sel(time=self.time).squeeze()
        self._ensure_has_coord(da=da, coord="xt")
        self._ensure_has_coord(da=da, coord="yt")

        if self.remove_gal_transform:
            tref = da_timedep.isel(time=0).time
            da = remove_gal_transform(da=da, tref=tref, base_name=self.base_name)

        if "longname" in da.attrs and not "long_name" in da.attrs:
            da.attrs["long_name"] = da.attrs["longname"]
            del da.attrs["longname"]

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da.to_netcdf(self.output().fn)

    def output(self):
        name_parts = [self.var_name, self.time.isoformat().replace(":", ""), "nc"]

        if self.remove_gal_transform:
            meta = _get_dataset_meta_info(self.base_name)
            u_gal, v_gal = meta["U_gal"]
            name_parts.insert(1, f"go_{u_gal}_{v_gal}")

        fn = ".".join(name_parts)

        p = get_workdir() / self.base_name / "cross_sections" / "runtime_slices" / fn
        return XArrayTarget(str(p))


class Extract2DCloudbaseStateFrom3D(luigi.Task):
    """
    Extract 2D field from 3D field near or below cloud-base from 3D dataset
    using `cldbase` 2D field. Where `cldbase` indicates the column contains no
    cloud the value will be nan
    """

    base_name = luigi.Parameter()
    field_name = luigi.Parameter()
    tn_3d = luigi.IntParameter()

    def requires(self):
        return dict(
            field=ExtractField3D(
                base_name=f"{self.base_name}.tn{self.tn_3d}", field_name=self.field_name
            ),
            cldbase=TimeCrossSectionSlices2D(
                base_name=self.base_name,
                var_name="cldbase",
            ),
        )

    @staticmethod
    def _extract_from_3d_at_heights_in_2d(da_3d, z_2d):
        z_unique = da_3d.sel(zt=slice(z_2d.min(), z_2d.max())).zt
        v = xr.concat(
            [da_3d.sel(zt=z_).where(z_2d == z_, other=np.nan) for z_ in tqdm(z_unique)],
            dim="zt",
        )
        return v.max(dim="zt", skipna=True)

    def run(self):
        da_cldbase_2d = self.input()["cldbase"].open()
        da_scalar_3d = self.input()["field"].open()

        dz = find_vertical_grid_spacing(da_scalar_3d)
        z_cb = da_cldbase_2d.sel(time=da_scalar_3d.time).squeeze()

        da_cb = self._extract_from_3d_at_heights_in_2d(
            da_3d=da_scalar_3d, z_2d=z_cb - dz
        )
        da_cb = da_cb.squeeze()
        da_cb.name = self.field_name
        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_cb.to_netcdf(self.output().fn)

    def output(self):
        fn = f"{self.base_name}.{self.field_name}.cldbase.tn{self.tn_3d}.xy.nc"
        p = get_workdir() / f"{self.base_name}.tn{self.tn_3d}" / "cross_sections" / fn
        return XArrayTarget(str(p))
