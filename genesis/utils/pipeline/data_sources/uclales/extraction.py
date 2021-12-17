"""
luigi-based pipeline for extracting full-domain 3D fields for single variables
at single timestep from per-core column output from the UCLALES model
"""
import signal
import subprocess
from pathlib import Path

import luigi
import xarray as xr

from ...data.base import XArrayTarget
from .common import _fix_time_units as fix_time_units

PARTIALS_3D_PATH = Path("partials_xr/3d")

SOURCE_BLOCK_FILENAME_FORMAT = "{file_prefix}.{i:04d}{j:04d}.nc"
SINGLE_VAR_BLOCK_FILENAME_FORMAT = "{file_prefix}.{i:04d}{j:04d}.{var_name}.tn{tn}.nc"
SINGLE_VAR_STRIP_FILENAME_FORMAT = "{file_prefix}.{dim}.{idx:04d}.{var_name}.tn{tn}.nc"
SINGLE_VAR_FILENAME_FORMAT = "{file_prefix}.{var_name}.tn{tn}.nc"

STORE_PARTIALS_LOCALLY = False


def _execute(cmd):
    print(" ".join(cmd))
    # https://stackoverflow.com/a/4417735
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()

    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def _call_cdo(args, verbose=True):
    try:
        cmd = ["cdo"] + args
        for output in _execute(cmd):
            if verbose:
                print((output.strip()))

    except subprocess.CalledProcessError as ex:
        return_code = ex.returncode
        error_extra = ""
        if -return_code == signal.SIGSEGV:
            error_extra = ", the utility segfaulted "

        raise Exception(
            "There was a problem when calling the tracking "
            "utility (errno={}): {} {}".format(error_extra, return_code, ex)
        )


class XArrayTargetUCLALES(XArrayTarget):
    def open(self, *args, **kwargs):
        kwargs["decode_times"] = False
        da = super().open(*args, **kwargs)
        da["time"], _ = fix_time_units(da["time"])
        if hasattr(da, "to_dataset"):
            return xr.decode_cf(da.to_dataset())
        else:
            return xr.decode_cf(da)


def _find_number_of_blocks(source_path, file_prefix):
    x_filename_pattern = SOURCE_BLOCK_FILENAME_FORMAT.format(
        file_prefix=file_prefix, i=9999, j=0
    ).replace("9999", "????")
    y_filename_pattern = SOURCE_BLOCK_FILENAME_FORMAT.format(
        file_prefix=file_prefix, j=9999, i=0
    ).replace("9999", "????")

    nx = len(list(Path(source_path).glob(x_filename_pattern)))
    ny = len(list(Path(source_path).glob(y_filename_pattern)))

    if nx == 0 or ny == 0:
        raise Exception(
            f"Didn't find any source files in `{source_path}` "
            f"(nx={nx} and ny={ny} found). Tried `{x_filename_pattern}` "
            f"and `{y_filename_pattern}` patterns"
        )

    return nx, ny


class UCLALESOutputBlock(luigi.ExternalTask):
    """
    Represents 3D output from model simulations
    """

    file_prefix = luigi.Parameter()
    source_path = luigi.Parameter()
    i = luigi.IntParameter()
    j = luigi.IntParameter()

    def output(self):
        fn = SOURCE_BLOCK_FILENAME_FORMAT.format(
            file_prefix=self.file_prefix, i=self.i, j=self.j
        )

        p = Path(self.source_path) / fn

        if not p.exists():
            raise Exception(f"Missing input file `{fn}` for `{self.file_prefix}`")

        return XArrayTargetUCLALES(str(p))


class UCLALESBlockSelectVariable(luigi.Task):
    """
    Extracts a single variable at a single timestep from one 3D output block

    {file_prefix}.{j:04d}{i:04d}.nc -> {file_prefix}.{j:04d}{i:04d}.{var_name}.tn{tn}.nc
    rico_gcss.00010002.nc -> rico_gcss.00010002.q.tn4.nc
    for var q and timestep 4
    """

    file_prefix = luigi.Parameter()
    source_path = luigi.Parameter()
    var_name = luigi.Parameter()
    i = luigi.IntParameter()
    j = luigi.IntParameter()
    tn = luigi.IntParameter()

    def requires(self):
        return UCLALESOutputBlock(
            file_prefix=self.file_prefix,
            i=self.i,
            j=self.j,
            source_path=self.source_path,
        )

    def run(self):
        ds_block = self.input().open()
        da_block_var = ds_block[self.var_name].isel(time=self.tn).expand_dims("time")

        # to use cdo the dimensions have to be (time, z, y, x)...
        posns = dict(time=0, zt=1, zm=1, yt=2, ym=2, xt=3, xm=3)
        dims = [None, None, None, None]
        for d in list(da_block_var.dims):
            dims[posns[d]] = d
        da_block_var = da_block_var.transpose(*dims)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_block_var.to_netcdf(self.output().fn)

    def output(self):
        fn = SINGLE_VAR_BLOCK_FILENAME_FORMAT.format(
            file_prefix=self.file_prefix,
            i=self.i,
            j=self.j,
            var_name=self.var_name,
            tn=self.tn,
        )
        p = Path(self.source_path) / PARTIALS_3D_PATH / fn

        return XArrayTargetUCLALES(str(p))


class UCLALESStripSelectVariable(luigi.Task):
    """
    Extracts a single variable at a single timestep as a strip of blocks along
    the `dim` dimension at index `idx` in the perpendicular dimension

    {file_prefix}.{j:04d}{i:04d}.nc -> {file_prefix}.{idx:04d}.{var_name}.tn{tn}.nc
    rico_gcss.00010002.nc -> rico_gcss.00010002.q.tn4.nc
    for var q and timestep 4
    """

    file_prefix = luigi.Parameter()
    source_path = luigi.Parameter()
    var_name = luigi.Parameter()
    idx = luigi.IntParameter()
    dim = luigi.Parameter()
    tn = luigi.IntParameter()
    use_cdo = luigi.BoolParameter(default=True)

    def requires(self):
        nx_b, ny_b = _find_number_of_blocks(
            file_prefix=self.file_prefix, source_path=self.source_path
        )

        if self.dim == "x":
            make_kws = lambda n: dict(i=self.idx, j=n)  # noqa
            nidx = ny_b
        elif self.dim == "y":
            make_kws = lambda n: dict(i=n, j=self.idx)  # noqa
            nidx = nx_b
        else:
            raise NotImplementedError(self.dim)

        return [
            UCLALESBlockSelectVariable(
                file_prefix=self.file_prefix,
                tn=self.tn,
                var_name=self.var_name,
                source_path=self.source_path,
                **make_kws(n=n),
            )
            for n in range(nidx)
        ]

    def _run_xarray(self):
        ortho_dim = "x" if self.dim == "y" else "y"

        dataarrays = [inp.open() for inp in self.input()]
        # x -> `xt` or `xm` mapping, similar for other dims
        da = dataarrays[0]
        dims = dict([(d.replace("t", "").replace("m", ""), d) for d in da.dims])

        ds_strip = xr.concat(dataarrays, dim=dims[ortho_dim])
        da_strip_var = ds_strip[self.var_name]
        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_strip_var.to_netcdf(self.output().fn)

    def run(self):
        if self.use_cdo:
            args = ["gather,1"] + [inp.fn for inp in self.input()] + [self.output().fn]
            _call_cdo(args)
        else:
            self._run_xarray()

    def output(self):
        fn = SINGLE_VAR_STRIP_FILENAME_FORMAT.format(
            file_prefix=self.file_prefix,
            idx=self.idx,
            dim=self.dim,
            var_name=self.var_name,
            tn=self.tn,
        )
        p = Path(self.source_path) / PARTIALS_3D_PATH / fn

        return XArrayTargetUCLALES(str(p))


class _Merge3DBaseTask(luigi.Task):
    """
    Common functionality for task that merge either strips or blocks together
    to construct datafile for whole domain
    """

    def requires(self):
        return dict(
            first_block=UCLALESBlockSelectVariable(
                file_prefix=self.file_prefix,
                i=0,
                j=0,
                var_name=self.var_name,
                tn=self.tn,
                source_path=self.source_path,
            )
        )

    def _check_output(self, da):
        # x -> `xt` or `xm` mapping, similar for other dims
        dims = dict([(d.replace("t", "").replace("m", ""), d) for d in da.dims])

        # check that we've aggregated enough bits and have the expected shape
        nx_b, ny_b = _find_number_of_blocks(
            file_prefix=self.file_prefix, source_path=self.source_path
        )
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

    def run(self):
        opened_inputs = dict([(inp, inp.open()) for inp in self.input()["parts"]])
        self._check_inputs(opened_inputs)
        da = xr.merge(opened_inputs.values())[self.var_name]
        self._check_output(da=da)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da.to_netcdf(self.output().fn)

    def _check_inputs(self, opened_inputs):
        pass

    def output(self):
        fn = SINGLE_VAR_FILENAME_FORMAT.format(
            file_prefix=self.file_prefix, tn=self.tn, var_name=self.var_name
        )
        p = Path(self.source_path) / PARTIALS_3D_PATH / fn
        return XArrayTarget(str(p))


class Extract3DbyBlocks(_Merge3DBaseTask):
    """
    Aggregate all nx*nx blocks for variable `var_name` at timestep `tn` into a
    single file
    """

    file_prefix = luigi.Parameter()
    source_path = luigi.Parameter()
    var_name = luigi.Parameter()
    tn = luigi.IntParameter()

    def requires(self):
        tasks = super().requires()
        nx, ny = _find_number_of_blocks(
            file_prefix=self.file_prefix, source_path=self.source_path
        )

        tasks_parts = []
        for i in range(nx):
            for j in range(ny):
                t = UCLALESBlockSelectVariable(
                    file_prefix=self.file_prefix,
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

    file_prefix = luigi.Parameter()
    source_path = luigi.Parameter()
    var_name = luigi.Parameter()
    tn = luigi.IntParameter()
    dim = luigi.Parameter(default="x")
    use_cdo = luigi.BoolParameter(default=True)

    def _check_inputs(self, opened_inputs):
        nx_b, ny_b = _find_number_of_blocks(
            file_prefix=self.file_prefix, source_path=self.source_path
        )

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
                invalid_shape[inp.fn] = strip_shape

        if len(invalid_shape) > 0:
            err_str = (
                "The following input strip files don't have the expected shape "
                f"{expected_shape}:\n\t"
            )

            err_str += "\n\t".join(
                [f"{shape}: {fn}" for (fn, shape) in invalid_shape.items()]
            )
            raise Exception(err_str)

    def run(self):
        if self.use_cdo:
            args = (
                ["gather"]
                + [inp.fn for inp in self.input()["parts"]]
                + [self.output().fn]
            )
            Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
            _call_cdo(args)
            # after running cdo we need to check it has the expected content
            da = self.output().open()
            try:
                self._check_output(da=da)
            except Exception:
                Path(self.output().fn).unlink()
                raise
        else:
            super(Extract3DbyStrips, self).run()

    def requires(self):
        nx, ny = _find_number_of_blocks(
            file_prefix=self.file_prefix, source_path=self.source_path
        )

        if self.dim == "x":
            nidx = nx
        elif self.dim == "y":
            nidx = ny
        else:
            raise NotImplementedError(self.dim)

        tasks = super().requires()

        tasks["parts"] = [
            UCLALESStripSelectVariable(
                file_prefix=self.file_prefix,
                dim=self.dim,
                idx=i,
                tn=self.tn,
                var_name=self.var_name,
                source_path=self.source_path,
            )
            for i in range(nidx)
        ]
        return tasks


class Extract3D(luigi.Task):
    file_prefix = luigi.Parameter()
    source_path = luigi.Parameter()
    var_name = luigi.Parameter()
    tn = luigi.IntParameter()
    mode = luigi.Parameter(default="y_strips")

    def requires(self):
        if self.mode == "blocks":
            return Extract3DbyBlocks(
                file_prefix=self.file_prefix,
                var_name=self.var_name,
                tn=self.tn,
                source_path=self.source_path,
            )
        elif self.mode.endswith("_strips"):
            return Extract3DbyStrips(
                file_prefix=self.file_prefix,
                var_name=self.var_name,
                tn=self.tn,
                dim=self.mode[0],
                source_path=self.source_path,
            )
        else:
            raise NotImplementedError(self.mode)

    def output(self):
        return self.input()
