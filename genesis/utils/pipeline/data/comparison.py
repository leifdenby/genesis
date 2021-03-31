"""
Facilitate creation of comparisons of data produced from data
pipeline-tasks called with different sets of parameters
"""
import importlib

import luigi


class Comparison(luigi.Task):
    base_class = luigi.Parameter()
    parameter_sets = luigi.DictParameter()
    global_parameters = luigi.DictParameter()

    def _get_task_class(self):
        k = self.base_class.rfind(".")
        class_module_name = "genesis.utils.pipeline.data"
        if k != -1:
            class_submodule_name, class_name = (
                self.base_class[:k],
                self.base_class[k + 1 :],
            )
            class_module_name = f"{class_module_name}.{class_submodule_name}"
        else:
            class_name = self.base_class

        class_module = importlib.import_module(class_module_name)
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
