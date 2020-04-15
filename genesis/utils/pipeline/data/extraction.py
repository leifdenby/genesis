import os
from pathlib import Path
import warnings
import importlib

import ipdb
import luigi
import xarray as xr
import numpy as np

from ....utils import calc_flux
from ... import mask_functions
from .base import get_workdir, XArrayTarget, _get_dataset_meta_info


if 'USE_SCHEDULER' in os.environ:
    from dask.distributed import Client
    client = Client(threads_per_worker=1)

COMPOSITE_FIELD_METHODS = dict(
    p_stddivs=(mask_functions.calc_scalar_perturbation_in_std_div, []),
    flux=(calc_flux.compute_vertical_flux, ['w', ]),
    _prefix__d=(calc_flux.get_horz_devition, []),
)


class ExtractField3D(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    FN_FORMAT = "{experiment_name}.{field_name}.nc"

    @staticmethod
    def _get_data_loader_module(meta):
        model_name = meta.get('model')
        if model_name is None:
            model_name = 'UCLALES'

        module_name = ".data_sources.{}".format(
            model_name.lower().replace('-', '_')
        )
        return importlib.import_module(module_name,
                                       package='genesis.utils.pipeline')

    def requires(self):
        meta = _get_dataset_meta_info(self.base_name)
        data_loader = self._get_data_loader_module(meta=meta)

        reqs = {}

        derived_fields = getattr(data_loader, 'DERIVED_FIELDS', None)

        if derived_fields is not None:
            for req_field in derived_fields.get(self.field_name, []):
                reqs[req_field] = ExtractField3D(base_name=self.base_name,
                                                 field_name=req_field)

        for (affix, (func, extra_fields)) in COMPOSITE_FIELD_METHODS.items():
            req_field = None
            if affix.startswith('_prefix__'):
                prefix = affix.replace('_prefix__', '')
                if self.field_name.startswith(prefix):
                    req_field = self.field_name.replace(
                        '{}_'.format(prefix), ''
                    )
            else:
                postfix = affix
                if self.field_name.endswith(postfix):
                    req_field = self.field_name.replace(
                        '_{}'.format(postfix), ''
                    )

            if req_field is not None:
                reqs['da'] = ExtractField3D(base_name=self.base_name,
                                            field_name=req_field)
                for v in extra_fields:
                    reqs[v] = ExtractField3D(base_name=self.base_name,
                                             field_name=v)

        return reqs

    def run(self):
        meta = _get_dataset_meta_info(self.base_name)

        fn_out = self.output()

        if fn_out.exists():
            pass
        elif meta['host'] == 'localhost':
            p_out = Path(self.output().fn)
            p_out.parent.mkdir(parents=True, exist_ok=True)

            is_composite = False
            for (affix, (func, _)) in COMPOSITE_FIELD_METHODS.items():
                if affix.startswith('_prefix__'):
                    prefix = affix.replace('_prefix__', '')
                    is_composite = self.field_name.startswith(prefix)
                else:
                    postfix = affix
                    is_composite = self.field_name.endswith(postfix)

                if is_composite:
                    das_input = dict([
                        (k, input.open(decode_times=False))
                        for (k, input) in self.input().items()
                    ])
                    with ipdb.launch_ipdb_on_exception():
                        da = func(**das_input)
                    # XXX: remove infs for now
                    da = da.where(~np.isinf(da))
                    da.to_netcdf(self.output().fn)
                    break

            if not is_composite:
                opened_inputs = dict([
                    (k, input.open()) for (k, input) in self.input().items()
                ])
                data_loader = self._get_data_loader_module(meta=meta)
                data_loader.extract_field_to_filename(
                    dataset_meta=meta, path_out=p_out,
                    field_name=self.field_name,
                    **opened_inputs
                )
        else:
            raise NotImplementedError(fn_out.fn)

    def output(self):
        meta = _get_dataset_meta_info(self.base_name)

        fn = self.FN_FORMAT.format(
            experiment_name=meta['experiment_name'],
            timestep=meta['timestep'],
            field_name=self.field_name
        )

        p = get_workdir()/self.base_name/fn

        t = XArrayTarget(str(p))

        if t.exists():
            data = t.open()
            if isinstance(data, xr.Dataset):
                if len(data.variables) == 0:
                    warnings.warn("Stored file for `{}` is empty, deleting..."
                                  "".format(self.field_name))
                    p.unlink()

        return t


class ExtractCrossSection2D(luigi.Task):
    base_name = luigi.Parameter()
    field_name = luigi.Parameter()

    FN_FORMAT = "{exp_name}.out.xy.{field_name}.nc"

    def _extract_and_symlink_local_file(self):
        meta = _get_dataset_meta_info(self.base_name)

        p_out = Path(self.output().fn)
        p_in = Path(meta['path'])/"cross_sections"/"runtime_slices"/p_out.name

        assert p_in.exists()

        p_out.parent.mkdir(exist_ok=True, parents=True)

        os.symlink(str(p_in.absolute()), str(p_out))

    def output(self):
        meta = _get_dataset_meta_info(self.base_name)

        fn = self.FN_FORMAT.format(
            exp_name=meta['experiment_name'],
            field_name=self.field_name
        )

        p = get_workdir()/self.base_name/"cross_sections"/"runtime_slices"/fn

        return luigi.LocalTarget(str(p))

    def run(self):
        meta = _get_dataset_meta_info(self.base_name)
        fn_out = self.output()

        if fn_out.exists():
            pass
        elif meta['host'] == 'localhost':
            self._extract_and_symlink_local_file()
        else:
            raise NotImplementedError(fn_out.fn)