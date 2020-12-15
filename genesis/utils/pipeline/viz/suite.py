import luigi


class Suite(luigi.Task):
    base_name = luigi.Parameter(default=None)
    timestep = luigi.IntParameter(default=None)

    DISTS = dict(
        mean_profile="mean.profiles",
        cross_section="cross.sections",
        cumulant_profiles="cumulant.profiles",
    )

    CROSS_SECTION_VARS = ["qv", "w", "qc", "cvrxp_p_stddivs"]

    def requires(self):
        reqs = {}
        if self.timestep is not None:

            def add_timestep(bn):
                return "{}.tn{}".format(bn, self.timestep)

        else:

            def add_timestep(bn):
                return bn

        if self.base_name is None:
            datasources = data.get_datasources()
            for base_name in datasources.keys():
                reqs["subsuite__{}".format(base_name)] = Suite(
                    base_name=add_timestep(base_name)
                )

            for v in Suite.CROSS_SECTION_VARS:
                base_names = ",".join([add_timestep(bn) for bn in datasources.keys()])
                reqs["cross_section__{}".format(v)] = CrossSection(
                    base_names=base_names, var_name=v, z="100.,400.,600.,800."
                )
                reqs["cumulant_profiles__{}".format(v)] = CumulantScalesProfile(
                    base_names=base_names,
                    cumulants="w:w,qv:qv,qc:qc,theta_l:theta_l,cvrxp:cvrxp,w:qv,w:qc,w:cvrxp",
                    z_max=1000.0,
                )
        else:
            reqs["mean_profile"] = HorizontalMeanProfile(
                base_name=self.base_name,
            )
        return reqs

    def run(self):
        for comp, target in self.input().items():
            if comp.startswith("subsuite__"):
                continue
            dst_path = self._build_output_path(comp=comp, target=target)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(target.fn, dst_path)

    def _build_output_path(self, comp, target):
        d = self.DISTS[comp.split("__")[0]]
        return Path(d) / target.fn

    def output(self):
        outputs = []
        for (comp, target) in self.input().items():
            if comp.startswith("subsuite__"):
                outputs.append(target)
            else:
                p = self._build_output_path(comp=comp, target=target)
                moved_target = luigi.LocalTarget(p)
                outputs.append(moved_target)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
