import os
import re
from datetime import datetime
from pathlib import Path

import dateutil.parser
import luigi
import numpy as np
import xarray as xr
import yaml

DATA_SOURCES = None
_WORKDIR = Path("data")


def add_datasource(name, attrs):
    global DATA_SOURCES
    if DATA_SOURCES is None:
        DATA_SOURCES = {}
    DATA_SOURCES[name] = attrs


def set_workdir(path):
    global _WORKDIR
    _WORKDIR = Path(path)


def get_workdir():
    return _WORKDIR


def get_datasources():
    if DATA_SOURCES is not None:
        datasources = DATA_SOURCES
    else:
        try:
            with open("datasources.yaml") as fh:
                loader = getattr(yaml, "FullLoader", yaml.Loader)
                datasources = yaml.load(fh, Loader=loader)
        except IOError:
            raise Exception("please define your data sources in" " datasources.yaml")

    return datasources


def _get_dataset_meta_info(base_name):
    datasources = get_datasources()

    datasource = None
    if datasources is not None:
        if base_name in datasources:
            datasource = datasources[base_name]
            if "tn" in base_name:
                _, timestep = base_name.split(".tn")
                datasource["timestep"] = int(timestep)
            else:
                datasource["timestep"] = 0
        elif re.search(r"\.tn[\d]+$", base_name):
            base_name, timestep = base_name.split(".tn")
            datasource = datasources[base_name]
            datasource["timestep"] = int(timestep)
        elif re.search(r"\.tn\*$", base_name):
            base_name, timestep = base_name.split(".tn")
            datasource = datasources[base_name]
            if timestep == "*":
                # this only happens when we're globbing to files, so we can
                # safely just set the timestep value without turning it into an
                # int
                datasource["timestep"] = timestep

    if datasource is None:
        raise Exception(
            "Please make a definition for `{}` in " "datasources.yaml".format(base_name)
        )
    else:
        # check the data path
        data_path = Path(datasource["path"])
        if data_path.exists():
            p_source_link = Path(get_workdir()) / "sources" / base_name
            if not p_source_link.exists():
                p_source_link.parent.mkdir(exist_ok=True, parents=True)
                os.symlink(datasource["path"], p_source_link)
        else:
            raise Exception(
                f"Source data path `{data_path}` for `{base_name}` not found"
            )

    return datasource


def get_data_from_task(task, local_scheduler=True, force_rerun=False):
    if force_rerun and task.output().exists():
        Path(task.output().fn).unlink()

    if not task.output().exists():
        luigi.build([task], local_scheduler=local_scheduler)
    if not task.output().exists():
        raise Exception("Something went wrong when processing the task")
    else:
        return task.output().open()


class XArrayTarget(luigi.target.FileSystemTarget):
    fs = luigi.local_target.LocalFileSystem()

    def __init__(self, path, *args, **kwargs):
        super(XArrayTarget, self).__init__(path, *args, **kwargs)
        self.path = path

    def open(self, *args, **kwargs):
        # ds = xr.open_dataset(self.path, engine='h5netcdf', *args, **kwargs)
        ds = xr.open_dataset(self.path, *args, **kwargs)

        if len(ds.data_vars) == 1:
            name = list(ds.data_vars)[0]
            da = ds[name]
            da.name = name
            return da
        else:
            return ds

    @property
    def fn(self):
        return self.path


class NumpyDatetimeParameter(luigi.DateSecondParameter):
    def normalize(self, x):
        if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.datetime64):
            dt64 = x
            # https://stackoverflow.com/a/13704307/271776
            ts = (dt64 - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
            return super().normalize(datetime.utcfromtimestamp(ts))
        else:
            try:
                return super().normalize(x)
            except TypeError:
                return super().normalize(dateutil.parser.parse(x))
