import pytest
import numpy as np
from scipy.constants import pi

from genesis.objects.topology.minkowski import analytical as minkowski_analytical
from genesis.objects.topology import shapes


@pytest.mark.parametrize("shape", ["cylinder", "spheroid", "ellipsoid"])
def test_reference_scales_calc(shape):
    kwargs = {}
    if shape == "ellipsoid":
        kwargs["lm"] = 0.5
    ds_scales = minkowski_analytical.calc_analytical_scales(shape=shape, **kwargs)

    # all the example shapes shown should have the same volume
    assert np.allclose(ds_scales.v0.max(), ds_scales.v0.min())

    assert "filamentarity" in ds_scales.data_vars


@pytest.mark.parametrize("shape", ["spheroid", "ellipsoid"])
def test_mink_scales_near_spherical(shape):
    """
    When the shape is a sphere the characteristic length, width and thickness
    should be equal to a sphere radius. So look for small deviation from a
    sphere to check implementation
    """
    l0 = 100.0
    dl = 1.0e-14

    kwargs = {}
    if shape == "spheroid":
        kwargs["a"] = l0 + dl
        kwargs["c"] = l0 - dl
    elif shape == "ellipsoid":
        kwargs["a"] = l0 + dl
        kwargs["b"] = l0
        kwargs["c"] = l0 - dl
    else:
        raise NotImplementedError(shape)

    A_approx = 4.0 * pi * l0 ** 2.0
    V_approx = 4.0 / 3.0 * pi * l0 ** 3.0

    fn = getattr(getattr(shapes, shape), "calc_minkowski_functionals")
    mink_functionals = fn(**kwargs)
    L, W, T = minkowski_analytical.length_scales(*mink_functionals)

    assert np.allclose(V_approx, mink_functionals[0])
    assert np.allclose(A_approx, mink_functionals[1] * 6.0)

    # for a sphere kappa1 = kappa2 = 1/r, so the integrated mean curvature is
    # 4*pi*r
    V2_approx = 1.0 / (3.0 * pi) * (4.0 * pi * l0)
    assert np.allclose(V2_approx, mink_functionals[2])
    assert mink_functionals[3] == 1
    assert np.allclose(L, W)
    assert np.allclose(W, T)


# volume and area (VA) and volume, area and curvature (VAC) for combinations of
# polar axis (a) and equitorial axis (c) values computed with
# https://rechneronline.de/pi/spheroid.php and from F. Gruy 2017 "Chord Length
# Distribution"
SPHEROID_SOLUTIONS = [
    # rechneronline.de:
    dict(
        ac=(20.0, 100.0),
        VA=(
            167551.608,
            20077.004,
        ),
    ),
    dict(
        ac=(100.0, 20.0),
        VA=(
            837758.041,
            68712.169,
        ),
    ),
    # F. Gruy. NB: the paper has swapped meaning of a and c axis, and curvature
    # value given is actually C/3 as the Minkowski functionals (with their
    # definition, missing the pi factor) is given
    dict(ac=(1.0, 2.0), VAC=(8.377, 21.48, 5.788 * 3.0)),
    dict(ac=(2.0, 1.0), VAC=(16.75, 34.69, 7.165 * 3.0)),
    dict(ac=(0.2, 2.0), VAC=(0.335, 3.966, 4.32 * 3.0)),
    dict(ac=(2.0, 0.2), VAC=(3.351, 25.89, 6.617 * 3.0)),
]


@pytest.mark.parametrize("shape", ["spheroid", "ellipsoid"])
@pytest.mark.parametrize("solution", SPHEROID_SOLUTIONS)
def test_mink_spheroid_solutions(shape, solution):
    kwargs = {}
    kwargs["a"], kwargs["c"] = solution["ac"]

    if shape == "ellipsoid":
        kwargs["b"] = kwargs["a"]

    fn = getattr(getattr(shapes, shape), "calc_minkowski_functionals")
    mink_functionals = fn(**kwargs)

    if "VA" in solution:
        V_sol, A_sol = solution["VA"]
        C_sol = None
    elif "VAC" in solution:
        V_sol, A_sol, C_sol = solution["VAC"]
    else:
        raise Exception

    V_calc = mink_functionals[0]
    A_calc = mink_functionals[1] * 6.0
    C_calc = mink_functionals[2] * 3.0 * pi

    assert np.allclose(V_calc, V_sol, rtol=1.0e-3)
    assert np.allclose(A_calc, A_sol, rtol=1.0e-3)
    if C_sol is not None:
        assert np.allclose(C_calc, C_sol, rtol=1.0e-2)


def test_mink_scales_across():
    """
    Check that the measures for the ellipsoid and spheroid are consistent
    """
    a = 100.0
    c = 200.0
    b = a
    # b = c = 100 - 1.0e-14

    mink_ellipsoid = shapes.ellipsoid.calc_minkowski_functionals(a=a, b=b, c=c)
    mink_spheroid = shapes.spheroid.calc_minkowski_functionals(a=a, c=c)

    for n in range(4):
        print(n)
        assert np.allclose(mink_ellipsoid[n], mink_spheroid[n])
