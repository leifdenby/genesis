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
SINGLE_VAR_STRIP_FILENAME_FORMAT = "{base_name}.{dim}.{idx:04d}.{var_name}.tn{tn}.nc"
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
    Extracts a single variable at a single timestep from one 3D output block

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
        meta = _get_dataset_meta_info(self.base_name)
        p = Path(meta["path"]) / RAW_DATA_PATH / PARTIALS_3D_PATH / fn

        return XArrayTarget(str(p))


class UCLALESStripSelectVariable(luigi.Task):
    """
    Extracts a single variable at a single timestep as a strip of blocks along
    the `dim` dimension at index `idx` in the perpendicular dimension

    {base_name}.{j:04d}{i:04d}.nc -> {base_name}.{idx:04d}.{var_name}.tn{tn}.nc
    rico_gcss.00010002.nc -> rico_gcss.00010002.q.tn4.nc
    for var q and timestep 4
    """

    base_name = luigi.Parameter()
    var_name = luigi.Parameter()
    idx = luigi.IntParameter()
    dim = luigi.Parameter()
    tn = luigi.IntParameter()

    def requires(self):
        nx_b, ny_b = _find_number_of_blocks(base_name=self.base_name)

        if self.dim == "x":
            make_kws = lambda n: dict(i=self.idx, j=n)  # noqa
            nidx = ny_b
        elif self.dim == "y":
            make_kws = lambda n: dict(i=n, j=self.idx)  # noqa
            nidx = nx_b
        else:
            raise NotImplementedError(self.dim)

        return [
            UCLALESOutputBlock(base_name=self.base_name, **make_kws(n=n))
            for n in range(nidx)
        ]

    def run(self):
        ortho_dim = "x" if self.dim == "y" else "y"

        dataarrays = [inp.open() for inp in self.input()]
        # x -> `xt` or `xm` mapping, similar for other dims
        da = dataarrays[0]
        dims = dict([(d.replace("t", "").replace("m", ""), d) for d in da.dims])

        ds_strip = xr.concat(dataarrays, dim=dims[ortho_dim])
        da_strip_var = ds_strip[self.var_name].isel(time=self.tn)
        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_strip_var.to_netcdf(self.output().fn)

    def output(self):
        fn = SINGLE_VAR_STRIP_FILENAME_FORMAT.format(
            base_name=self.base_name,
            idx=self.idx,
            dim=self.dim,
            var_name=self.var_name,
            tn=self.tn,
        )
        meta = _get_dataset_meta_info(self.base_name)
        p = Path(meta["path"]) / RAW_DATA_PATH / PARTIALS_3D_PATH / fn

        return XArrayTarget(str(p))


class _Merge3DBaseTask(luigi.Task):
    """
    Common functionality for task that merge either strips or blocks together
    to construct datafile for whole domain
    """

    def requires(self):
        return dict(
            first_block=UCLALESBlockSelectVariable(
                base_name=self.base_name,
                i=0,
                j=0,
                var_name=self.var_name,
                tn=self.tn,
            )
        )

    def run(self):
        opened_inputs = dict([(inp, inp.open()) for inp in self.input()["parts"]])
        self._check_inputs(opened_inputs)
        da = xr.merge(opened_inputs.values())[self.var_name]
        # x -> `xt` or `xm` mapping, similar for other dims
        dims = dict([(d.replace("t", "").replace("m", ""), d) for d in da.dims])

        # check that we've aggregated enough bits and have the expected shape
        nx_b, ny_b = _find_number_of_blocks(base_name=self.base_name)
        da_first_block = self.input()["first_block"].open()
        b_nx = int(da_first_block[dims["x"]].count())
        b_ny = int(da_first_block[dims["y"]].count())

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

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da.to_netcdf(self.output().fn)

    def _check_inputs(self, opened_inputs):
        pass

    def output(self):
        fn = SINGLE_VAR_FILENAME_FORMAT.format(
            base_name=self.base_name, tn=self.tn, var_name=self.var_name
        )
        p = get_workdir() / OUTPUT_DATA_PATH / fn
        return XArrayTarget(str(p))


class Extract3DbyBlocks(_Merge3DBaseTask):
    """
    Aggregate all nx*nx blocks for variable `var_name` at timestep `tn` into a
    single file
    """

    base_name = luigi.Parameter()
    var_name = luigi.Parameter()
    tn = luigi.IntParameter()

    def requires(self):
        tasks = super().requires()
        nx, ny = _find_number_of_blocks(base_name=self.base_name)

        tasks_parts = []
        for i in range(nx):
            for j in range(ny):
                t = UCLALESBlockSelectVariable(
                    base_name=self.base_name,
                    var_name=self.var_name,
                    i=i,
                    j=j,
                    tn=self.tn,
                )
                tasks_parts.append(t)

        tasks["parts"] = tasks_parts
        return tasks


class Extract3DbyStrips(_Merge3DBaseTask):
    """
    Aggregate all strips along `dim` dimension for `var_name` at timestep `tn` into a
    single file
    """

    base_name = luigi.Parameter()
    var_name = luigi.Parameter()
    tn = luigi.IntParameter()
    dim = luigi.Parameter(default="x")

    def _check_inputs(self, opened_inputs):
        nx_b, ny_b = _find_number_of_blocks(base_name=self.base_name)

        # find block size
        da_first_block = self.input()["first_block"].open()
        dims = dict(
            [(d.replace("t", "").replace("m", ""), d) for d in da_first_block.dims]
        )
        b_nx = int(da_first_block[dims["x"]].count())
        b_ny = int(da_first_block[dims["y"]].count())

        if self.dim == "x":
            expected_shape = (b_nx, b_ny * ny_b)
        elif self.dim == "y":
            expected_shape = (b_nx * nx_b, b_ny)

        invalid_shape = {}
        for inp, da_strip in opened_inputs.items():
            strip_shape = (
                int(da_strip[dims["x"]].count()),
                int(da_strip[dims["y"]].count()),
            )
            if strip_shape != expected_shape:
                invalid_shape[inp.fn] = da_strip.shape

        if len(invalid_shape) > 0:
            err_str = (
                "The following input strip files don't have the expected shape "
                f"{expected_shape}:\n"
            )

            err_str += "\n\t".join(
                [f"{shape}: {fn}" for (fn, shape) in invalid_shape.items()]
            )
            raise Exception(err_str)

    def requires(self):
        nx, ny = _find_number_of_blocks(base_name=self.base_name)

        if self.dim == "x":
            nidx = nx
        elif self.dim == "y":
            nidx = ny
        else:
            raise NotImplementedError(self.dim)

        tasks = super().requires()

        tasks["parts"] = [
            UCLALESStripSelectVariable(
                base_name=self.base_name,
                dim=self.dim,
                idx=i,
                tn=self.tn,
                var_name=self.var_name,
            )
            for i in range(nidx)
        ]
        return tasks


class Extract3D(luigi.WrapperTask):
    base_name = luigi.Parameter()
    var_name = luigi.Parameter()
    tn = luigi.IntParameter()
    mode = luigi.Parameter(default="x_strips")

    def requires(self):
        if self.mode == "blocks":
            return Extract3DbyBlocks(
                base_name=self.base_name, var_name=self.var_name, tn=self.tn
            )
        elif self.mode.endswith("_strips"):
            return Extract3DbyStrips(
                base_name=self.base_name,
                var_name=self.var_name,
                tn=self.tn,
                dim=self.mode[0],
            )
        else:
            raise NotImplementedError(self.mode)
