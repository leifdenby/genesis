from pathlib import Path
import luigi
import xarray as xr

from .base import (
    get_workdir,
    XArrayTarget,
    _get_dataset_meta_info,
)
from ..data_sources.uclales import _fix_time_units as fix_time_units

RAW_DATA_PATH = Path("raw_data")
OUTPUT_DATA_PATH = Path("3d_blocks/full_domain")
PARTIALS_3D_PATH = Path("partials_xr/3d")

SOURCE_BLOCK_FILENAME_FORMAT = "{base_name}.{i:04d}{j:04d}.nc"
SINGLE_VAR_BLOCK_FILENAME_FORMAT = "{base_name}.{i:04d}{j:04d}.{var_name}.tn{tn}.nc"
SINGLE_VAR_FILENAME_FORMAT = "{base_name}.{var_name}.tn{tn}.nc"


class XArrayTargetUCLALES(XArrayTarget):
    def open(self, *args, **kwargs):
        kwargs["decode_times"] = False
        da = super().open(*args, **kwargs)
        da["time"], _ = fix_time_units(da["time"])
        if hasattr(da, "to_dataset"):
            return xr.decode_cf(da.to_dataset())
        else:
            return xr.decode_cf(da)


def _find_number_of_blocks(base_name):
    meta = _get_dataset_meta_info(base_name)
    source_path = Path(meta["path"]) / RAW_DATA_PATH

    x_filename_pattern = SOURCE_BLOCK_FILENAME_FORMAT.format(
        base_name=base_name, i=9999, j=0
    ).replace("9999", "????")
    y_filename_pattern = SOURCE_BLOCK_FILENAME_FORMAT.format(
        base_name=base_name, j=9999, i=0
    ).replace("9999", "????")

    nx = len(list(source_path.glob(x_filename_pattern)))
    ny = len(list(source_path.glob(y_filename_pattern)))

    if nx == 0 or ny == 0:
        raise Exception(
            f"Didn't find any source files in `{source_path}` "
            f"(nx={nx} and ny={ny} found)"
        )

    return nx, ny


class UCLALESOutputBlock(luigi.ExternalTask):
    """
    Represents 3D output from model simulations
    """

    base_name = luigi.Parameter()
    i = luigi.IntParameter()
    j = luigi.IntParameter()

    def output(self):
        fn = SOURCE_BLOCK_FILENAME_FORMAT.format(
            base_name=self.base_name, i=self.i, j=self.j
        )

        meta = _get_dataset_meta_info(self.base_name)
        p = Path(meta["path"]) / RAW_DATA_PATH / fn

        if not p.exists():
            raise Exception(f"Missing input file `{fn}` for `{self.base_name}`")

        return XArrayTargetUCLALES(str(p))


class UCLALESBlockSelectVariable(luigi.Task):
    """
    {base_name}.{j:04d}{i:04d}.nc -> {base_name}.{j:04d}{i:04d}.{var_name}.tn{tn}.nc
    rico_gcss.00010002.nc -> rico_gcss.00010002.q.tn4.nc
    for var q and timestep 4
    """

    base_name = luigi.Parameter()
    var_name = luigi.Parameter()
    i = luigi.IntParameter()
    j = luigi.IntParameter()
    tn = luigi.IntParameter()

    def requires(self):
        return UCLALESOutputBlock(base_name=self.base_name, i=self.i, j=self.j)

    def run(self):
        ds_block = self.input().open()
        da_block_var = ds_block[self.var_name].isel(time=self.tn)
        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_block_var.to_netcdf(self.output().fn)

    def output(self):
        fn = SINGLE_VAR_BLOCK_FILENAME_FORMAT.format(
            base_name=self.base_name,
            i=self.i,
            j=self.j,
            var_name=self.var_name,
            tn=self.tn,
        )
        p = get_workdir() / OUTPUT_DATA_PATH / fn

        return XArrayTarget(str(p))


class Extract3D_UCLALES(luigi.Task):
    base_name = luigi.Parameter()
    var_name = luigi.Parameter()
    tn = luigi.IntParameter()

    def requires(self):
        tasks = []
        nx, ny = _find_number_of_blocks(base_name=self.base_name)

        for i in range(nx):
            for j in range(ny):
                t = UCLALESBlockSelectVariable(
                    base_name=self.base_name,
                    var_name=self.var_name,
                    i=i,
                    j=j,
                    tn=self.tn,
                )
                tasks.append(t)
        return tasks

    def run(self):
        da = xr.merge([inp.open() for inp in self.input()])[self.var_name]
        # x -> `xt` or `xm` mapping, similar for other dims
        dims = dict([(d.replace("t", "").replace("m", ""), d) for d in da.dims])

        # check that we've aggregated enough bits and have the expected shape
        nx_b, ny_b = _find_number_of_blocks(base_name=self.base_name)
        da_first_block = self.input()[0].open()
        b_nx = da_first_block[dims["x"]].count()
        b_ny = da_first_block[dims["y"]].count()

        nx_da = int(da.coords[dims["x"]].count())
        ny_da = int(da.coords[dims["y"]].count())

        if nx_da != (b_nx * nx_b):
            raise Exception(
                "Resulting data is the the wrong size " f"( {nx_da} != {b_nx} x {nx_b})"
            )

        if ny_da != (b_ny * ny_b):
            raise Exception(
                "Resulting data is the the wrong size " f"( {ny_da} != {b_ny} x {ny_b})"
            )

        da.to_netcdf(self.output().fn)

    def output(self):
        fn = SINGLE_VAR_FILENAME_FORMAT.format(
            base_name=self.base_name, tn=self.tn, var_name=self.var_name
        )
        p = get_workdir() / OUTPUT_DATA_PATH / fn

        return XArrayTarget(str(p))
