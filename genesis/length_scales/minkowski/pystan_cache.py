"""
create a local cache for pyStan models, based on
https://pystan.readthedocs.io/en/latest/avoiding_recompilation.html#automatically-reusing-models
"""
import pystan
import pickle
from hashlib import md5
from pathlib import Path


def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)

    # save the cached model relative to this script file
    p = Path(__file__).parent/cache_fn
    cache_fn = str(p)

    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    return sm
