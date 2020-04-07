import luigi

from . import viz

class FluxFractionCarriedSummary(viz.FluxFractionCarried):
    z_pad = luigi.FloatParameter(default=100)

    def run(self):
        super().run()

        input = self.input()
        ds = self.input().open()

        name = "{}_flux__mean".format(self.scalar)

        def scale_flux(da_flux):
            if da_flux.sampling == 'full domain':
                return da_flux
            else:
                return da_flux*ds.areafrac.sel(sampling=da_flux.sampling)
        da_flux_tot = ds[name].groupby('sampling').apply(scale_flux)

        z_lims = self.z_pad, self.z_max-self.z_pad
        da_flux_tot_range = da_flux_tot.sel(
            zt=slice(*z_lims)
        )

        captured_fraction = (
            da_flux_tot_range.sel(sampling='objects')
            /da_flux_tot_range.sel(sampling='mask')
        ).mean(dim='zt')

        with open(self.output()['txt'].fn, 'w') as fh:
            fh.write("mean fraction of mask flux carried by objects"
                    " between {}m and {}m: {:.2f}%".format(
                *z_lims, 100.*captured_fraction.values
            ))

    def output(self):
        output = super().output()
        fn_txt = output['plot'].fn.replace('.png', '.txt')
        output['txt'] = luigi.LocalTarget(fn_txt)
        return output
