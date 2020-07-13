import pytest


from genesis.objects.topology import minkowski_analytical


@pytest.mark.parametrize("shape", ["cylinder", "spheroid", "ellipsoid"])
def test_reference_scales_calc(shape):
    kwargs = {}
    if shape == "ellipsoid":
        kwargs['alpha'] = 0.5
    ds_scales = minkowski_analytical.calc_analytical_scales(shape=shape, **kwargs)

    assert 'filamentarity' in ds_scales.data_vars
