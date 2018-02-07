import os

import xarray as xr

DATA_ROOT = '/nfs/see-fs-02_users/earlcd/datastore/a289/LES_datasets/'

def get_fn(model_name, case_name, param_name, var_name, tn):
    return os.path.join(DATA_ROOT, model_name, case_name, param_name, 
                        '{}.tn{}.{}.nc'.format(case_name, tn, var_name))

def get_data(*args, **kwargs):
    fn = get_fn(*args, **kwargs)
    return xr.open_dataset(fn, decode_times=False)
