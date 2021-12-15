import subprocess
from pathlib import Path

import luigi


class PlotJoinTask(luigi.Task):
    """
    Convenience base task which calls imagemagick to join multiple output
    images into a single image, either horizontally or vertically. Implement
    `self._build_tasks` to return dictionary with keys for the names of the
    joined plots and the tasks that produce the plots to join.
    """

    direction = luigi.Parameter(default="horizontal")

    def requires(self):
        return self._build_tasks()

    def _build_tasks(self, return_plotjoins=False):
        raise NotImplementedError

    def run(self):
        plotjoins = self._build_tasks(return_plotjoins=True)

        for join_name, tasks in plotjoins.items():
            Path(self.output()[join_name].fn).parent.mkdir(exist_ok=True, parents=True)
            args = [
                "convert",
                *[t.output().fn for t in tasks],
                "{}append".format(["-", "+"][self.direction == "horizontal"]),
                self.output()[join_name].fn,
            ]
            subprocess.call(args)

    def output(self):
        plotjoins = self._build_tasks(return_plotjoins=True)

        return {
            join_name: luigi.LocalTarget("joined_plots/{}.png".format(join_name))
            for join_name in plotjoins.keys()
        }
