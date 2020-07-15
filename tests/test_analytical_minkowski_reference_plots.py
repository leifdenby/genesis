import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
from genesis.objects.topology.plots.filamentarity_planarity import plot_reference  # noqa


@pytest.mark.parametrize("shape", ["cylinder", "spheroid"])
def test_plot_reference(shape):
    kwargs = {}
    if shape == "ellipsoid":
        kwargs["lm"] = 0.5
    _, ax = plt.subplots()
    _ = plot_reference(ax=ax, shape=shape, **kwargs)
