import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

from genesis.objects.topology.plots.filamentarity_planarity import \
    plot_reference  # noqa


@pytest.mark.parametrize("shape", ["cylinder", "spheroid", "ellipsoid"])
def test_plot_reference(shape):
    calc_kwargs = dict(N_points=20)  # plot few points to make the test faster
    _, ax = plt.subplots()
    _ = plot_reference(ax=ax, shape=shape, calc_kwargs=calc_kwargs)
