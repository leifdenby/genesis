
# coding: utf-8

import numpy as np
from scipy.constants import pi
import xarray as xr

def _f2(x):
    return np.arccos(x)/np.sqrt(1 - x**2.)

def _spheroid_minkowski(r, lm):
    @np.vectorize
    def f(x):
        # http://www.suitcaseofdreams.net/inverse_functions.htm#P1
        d = np.sqrt(1 + 0J - x**2.)

        if d.imag > 0.0:
            assert np.all(d.real == 0.0)
            return np.log(x + np.sqrt(x**2. - 1.))/d.imag
        else:
            return np.arccos(x)/d.real

    V0 = 4.*pi/3.*r**3.*lm
    V1 = pi/3.*r**2.*(1+f(1./lm))
    V2 = 2./3.*r*(lm+f(lm))
    V3 = 1.
    
    return V0, V1, V2, V3

@np.vectorize
def _spheroid_minkowski2(r, lm):
    @np.vectorize
    def f(x):
        # http://www.suitcaseofdreams.net/inverse_functions.htm#P1
        d = np.sqrt(1 + 0J - x**2.)

        if d.imag > 0.0:
            assert np.all(d.real == 0.0)
            return np.log(x + np.sqrt(x**2. - 1.))/d.imag
        else:
            return np.arccos(x)/d.real

    V0 = 4.*pi/3.*r**3.*lm
    
    if lm > 1.0:
        e = np.sqrt(1 - (1./lm)**2.)
        V1 = 2./6.*pi*r**2.*(1 + lm/e*np.arcsin(e))
    else:
        e = np.sqrt(1 - lm**2.)
        V1 = 2./6.*pi*r**2.*(1 + (1 - e**2.)/e*np.arctanh(e))
    
    V2 = 2./3.*r*(lm+f(lm))
    V3 = 1.
    
    return V0, V1, V2, V3

def _cylinder_minkowski(r, lm):
    V0 = pi*r**3.*lm
    V1 = pi/3.*r**2.*(1+lm)
    V2 = 1./3.*r*(pi+lm)
    V3 = 1.
    
    return V0, V1, V2, V3

def length_scales(fn_mink, r, lm):
    V0, V1, V2, V3 = fn_mink(r=r, lm=lm)
    
    T = V0/(2.*V1)
    W = 2*V1/(pi*V2)
    L = 3*V2/(4*V3)
    
    return L, W, T
    
def filamentarity_planarity(fn_mink, r, lm):
    L, W, T = length_scales(fn_mink=fn_mink, r=r, lm=lm)
    
    P = (W-T)/(W+T)
    F = (L-W)/(L+W)
    
    return F, P

def calc_analytical_scales(shape):
    r_ = 100.
    m_min, m_max = -6, 6
    m = np.linspace(-6., 6., (m_max - m_min)*10+1)
    i = np.arange(len(m))
    lm = 2.**m
    r = np.array([r_,]*len(lm))

    if shape == "cylinder":
        fn_mink=_cylinder_minkowski
    elif shape == "spheroid":
        fn_mink=_spheroid_minkowski2
    else:
        raise NotImplementedError

    LWT = length_scales(fn_mink=_cylinder_minkowski, r=r, lm=lm)
    FP = filamentarity_planarity(fn_mink=fn_mink, r=r, lm=lm)

    ds = xr.Dataset(coords=dict(i=i))
    for n, label in enumerate("length width thinckness".split(" ")):
        ds[label] = (('i',), LWT[n])
        ds[label].attrs['units'] = "1"

    ds['filamentarity'] = (('i',), FP[0])
    ds['planarity'] = (('i',), FP[1])
    ds['lm'] = (('i',), lm)

    return ds
