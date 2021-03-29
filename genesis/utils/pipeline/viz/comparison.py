"""
Facilitate creation of comparison plots of data produced from data
pipeline-tasks called with different sets of parameters
"""
import importlib
import matplotlib.pyplot as plt
import xarray as xr
import luigi
import seaborn as sns

from ..data.objects import ObjectTwoScalesComposition


def _apply_plot_parameters_to_axis(ax, plot_parameters):
    if plot_parameters.get("despine"):
        sns.despine(ax=ax)

    if plot_parameters.get("x_max"):
        ax.set_xlim(0, plot_parameters["x_max"])

    if plot_parameters.get("y_max"):
        ax.set_ylim(0, plot_parameters["y_max"])

    title = plot_parameters.get("title", "")
    ax.set_title(title)


def _scales_dist_2d(datasets, plot_parameters):
    fig, ax = plt.subplots()

    cmaps = ["Blues", "Reds", "Greens", "Oranges"]

    for (cmap, (task_name, da)) in zip(cmaps, datasets.items()):
        # da = ds.sel(task_name=task_name)
        cs = da.plot.contour(ax=ax, cmap=cmap)
        line = cs.collections[2]
        line.set_label(task_name)

    _apply_plot_parameters_to_axis(ax=ax, plot_parameters=plot_parameters)

    ax.legend()
    return fig


class ComparisonPlot(luigi.Task):
    base_class = luigi.Parameter()
    parameter_sets = luigi.DictParameter()
    global_parameters = luigi.DictParameter()
    plot_parameters = luigi.DictParameter()
    name = luigi.Parameter()

    def _get_task_class(self):
        k = self.base_class.rfind(".")
        class_module_name, class_name = self.base_class[:k], self.base_class[k + 1 :]
        class_module = importlib.import_module(
            f"genesis.utils.pipeline.data.{class_module_name}"
        )
        TaskClass = getattr(class_module, class_name)
        return TaskClass

    def requires(self):
        TaskClass = self._get_task_class()

        tasks = {}
        for task_name, parameter_set in self.parameter_sets.items():
            task_parameters = dict(self.global_parameters)
            task_parameters.update(parameter_set)
            t = TaskClass(**task_parameters)
            tasks[task_name] = t

        return tasks

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
            datasets[task_name] = ds

        # ds = xr.concat(datasets, dim="task_name")

        fig = plot_function(datasets=datasets, plot_parameters=self.plot_parameters)

        plt.savefig(self.output().fn, fig=fig)

    def output(self):
        fn = f"{self.name}.png"
        return luigi.LocalTarget(fn)
