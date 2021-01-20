import luigi
import xarray as xr
import matplotlib.pyplot as plt


from .... import length_scales
from .. import data


class CumulantSlices(luigi.Task):
    v1 = luigi.Parameter()
    v2 = luigi.Parameter()
    base_names = luigi.Parameter()

    z_step = luigi.IntParameter(default=4)
    z_max = luigi.FloatParameter(default=700.0)
    filetype = luigi.Parameter(default="pdf")

    def requires(self):
        base_names = self.base_names.split(",")

        return dict(
            (
                base_name,
                {
                    self.v1: data.ExtractField3D(
                        field_name=self.v1, base_name=base_name
                    ),
                    self.v2: data.ExtractField3D(
                        field_name=self.v2, base_name=base_name
                    ),
                },
            )
            for base_name in base_names
        )

    def get_suptitle(self, base_name):
        return base_name

    def makeplot(self):
        base_names = self.base_names.split(",")

        datasets = []

        for base_name in base_names:
            inputs = self.input()[base_name]
            v1 = inputs[self.v1].open(decode_times=False)
            v2 = inputs[self.v2].open(decode_times=False)

            ds = xr.merge([v1, v2])
            ds = ds.sel(zt=slice(0.0, self.z_max))
            ds = ds.isel(zt=slice(None, None, self.z_step))
            ds.attrs["name"] = self.get_suptitle(base_name)
            datasets.append(ds)

        plot_fn = length_scales.cumulant.sections.plot
        import ipdb

        with ipdb.launch_ipdb_on_exception():
            ax = plot_fn(
                datasets=datasets,
                var_names=[self.v1, self.v2],
            )

        return ax

    def run(self):
        self.makeplot()
        plt.savefig(self.output().fn, bbox_inches="tight")

    def output(self):
        fn = length_scales.cumulant.sections.FN_FORMAT_PLOT.format(
            v1=self.v1, v2=self.v2, filetype=self.filetype
        )
        base_name = self.base_names.replace(",", "__")

        fn = f"{base_name}.{fn}"

        return luigi.LocalTarget(fn)


class CumulantScalesProfile(luigi.Task):
    base_names = luigi.Parameter()
    cumulants = luigi.Parameter()
    z_max = luigi.FloatParameter(default=700.0)
    plot_type = luigi.Parameter(default="scales")
    filetype = luigi.Parameter(default="pdf")
    scale_limits = luigi.Parameter(default="")

    mask = luigi.Parameter(default=None)
    mask_args = luigi.Parameter(default="")

    def _parse_cumulant_arg(self):
        cums = [c.split(":") for c in self.cumulants.split(",")]
        return [c for (n, c) in enumerate(cums) if cums.index(c) == n]

    def _parse_scale_limits(self):
        """
        format: `C(w,w)=100.0;C(q,q)=2000.0`
        """

        def _make_item(l):
            if "=" in l:
                k, v = l.split("=")
                return (k, float(v))

        limits = {}
        for l in self.scale_limits.split(":"):
            item = _make_item(l)
            if item:
                k, v = item
                limits[k] = v

        return limits

    def requires(self):
        return data.ExtractCumulantScaleProfiles(
            base_names=self.base_names,
            cumulants=self.cumulants,
            mask=self.mask,
            mask_args=self.mask_args,
            z_max=self.z_max,
        )

    def run(self):
        ds = self.input().open()

        cumulants = self._parse_cumulant_arg()
        cumulants_s = ["C({},{})".format(c[0], c[1]) for c in cumulants]

        plot_fn = length_scales.cumulant.vertical_profile.plot.plot

        plot_fn(
            data=ds,
            cumulants=cumulants_s,
            plot_type=self.plot_type,
            scale_limits=self._parse_scale_limits(),
        )

        plt.savefig(self.output().path, bbox_inches="tight")

    def output(self):
        base_name = "__".join(self.base_names.split(","))
        fn = length_scales.cumulant.vertical_profile.plot.FN_FORMAT.format(
            base_name=base_name,
            plot_type=self.plot_type,
            mask=self.mask or "no_mask",
            filetype=self.filetype,
        )
        return luigi.LocalTarget(fn)
