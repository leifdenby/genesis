"""
Facilitate creation of comparison plots of data produced from data
pipeline-tasks called with different sets of parameters
"""
import textwrap
import warnings

import luigi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ....objects.topology.plots.filamentarity_planarity import \
    plot_reference as fp_reference
from ...plot_types import PlotGrid, get_color_cmap
from ..data import Comparison
from ..data.objects import ObjectTwoScalesComposition


def _generate_title_from_global_parameters(global_parameters):
    s = ", ".join(f"{k}: {v}" for (k, v) in global_parameters.items())
    return "\n".join(
        textwrap.wrap(
            s,
            width=100,
        )
    )


def _apply_plot_parameters_to_axis(
    ax, plot_parameters, global_parameters, use_suptitle=False
):
    if plot_parameters.get("despine"):
        sns.despine(ax=ax)

    if plot_parameters.get("x_max"):
        ax.set_xlim(0, plot_parameters["x_max"])

    if plot_parameters.get("y_max"):
        ax.set_ylim(0, plot_parameters["y_max"])

    title = plot_parameters.get(
        "title", _generate_title_from_global_parameters(global_parameters)
    )
    if use_suptitle:
        ax.figure.suptitle(title)
        ax.set_title("")
    else:
        ax.set_title(title)


def _scales_dist_2d(datasets, plot_parameters, global_parameters):
    add_marginal_distributions = plot_parameters.get(
        "add_marginal_distributions", False
    )
    add_scatter = plot_parameters.get("add_scatter", False)

    fig_params = plot_parameters.get("fig_params", {})
    if add_marginal_distributions:
        pg_kwargs = {}
        if "figsize" in fig_params:
            pg_kwargs["height"] = fig_params["figsize"][1]
        g = PlotGrid(**pg_kwargs)
        ax_joint = g.ax_joint
        fig = g.fig
    else:
        fig, ax_joint = plt.subplots(**fig_params)

    annotations = plot_parameters.get("annotations", [])
    if "filamentarity_planarity_reference" in annotations:
        try:
            kwargs = dict(annotations["filamentarity_planarity_reference"])
        except KeyError:
            kwargs = {}
        fp_reference(ax=ax_joint, shape="spheroid", color="black", **kwargs)

    if "unit_line" in annotations:
        if not hasattr(ax_joint, "axline"):
            warnings.warn("Upgrade to matplotlib >= 3.3.0 to draw unit line")
        else:
            ax_joint.axline(
                xy1=(0.0, 0.0), xy2=(1.0, 1.0), linestyle="--", color="grey"
            )

    if "line_colors" in plot_parameters:
        cmaps = [get_color_cmap(color=c) for c in plot_parameters["line_colors"]]
    else:
        cmaps = ["Blues", "Reds", "Greens", "Oranges"]

    for (cmap, (task_name, (task, da))) in zip(cmaps, datasets.items()):
        # da = ds.sel(task_name=task_name)
        cs = da.plot.contour(ax=ax_joint, cmap=cmap)
        line = cs.collections[2]
        line.set_label(task_name)

        if add_marginal_distributions:
            da.sum(dim=da.dims[0], dtype=np.float64, keep_attrs=True).plot(
                ax=g.ax_marg_x,
                color=line.get_color(),
                drawstyle="steps-mid",
            )
            da.sum(dim=da.dims[1], dtype=np.float64, keep_attrs=True).plot(
                ax=g.ax_marg_y,
                color=line.get_color(),
                drawstyle="steps-mid",
                y=da.dims[0],
            )

        if add_scatter:
            # to create a scatter plot we need the parent task which computed the scales
            ds_scales = task.requires()["scales"].output().open()
            ax_joint.scatter(
                ds_scales[da.dims[1]],
                ds_scales[da.dims[0]],
                marker=".",
                s=1.0,
                alpha=plot_parameters.get("scatter_alpha", 0.4),
                color=line.get_color(),
            )

    if add_marginal_distributions:

        def remove_axes_annotations(ax_):
            ax_.set_title("")
            ax_.set_xlabel("")
            ax_.set_ylabel("")

        remove_axes_annotations(g.ax_marg_x)
        remove_axes_annotations(g.ax_marg_y)
        g.ax_marg_y.set_xticklabels([])
        g.ax_marg_x.set_yticklabels([])

    _apply_plot_parameters_to_axis(
        ax=ax_joint,
        plot_parameters=plot_parameters,
        use_suptitle=add_marginal_distributions,
        global_parameters=global_parameters,
    )

    ax_joint.legend(loc=plot_parameters.get("legend_loc", "best"))
    plt.tight_layout()
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


class ComparisonPlot(Comparison):
    """
    Special type of data comparison which produces a plot
    """

    plot_parameters = luigi.DictParameter()
    output_filename = luigi.Parameter()

    def run(self):
        TaskClass = self._get_task_class()

        if TaskClass == ObjectTwoScalesComposition:
            plot_function = _scales_dist_2d
        else:
            raise NotImplementedError(TaskClass)

        datasets = {}
        for task_name, task_input in self.input().items():
            ds = task_input.open()
            ds["task_name"] = task_name
            datasets[task_name] = (self.requires()[task_name], ds)

        # ds = xr.concat(datasets, dim="task_name")

        fig = plot_function(
            datasets=datasets,
            plot_parameters=self.plot_parameters,
            global_parameters=self.global_parameters,
        )

        plt.savefig(self.output().fn, fig=fig, bbox_inches="tight")

    def output(self):
        return luigi.LocalTarget(self.output_filename)
