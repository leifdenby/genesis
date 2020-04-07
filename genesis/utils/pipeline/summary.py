import luigi
import numpy as np

from . import viz, data

class FluxFractionCarriedSummary(viz.FluxFractionCarried):
    z_pad = luigi.FloatParameter(default=100)

    def requires(self):
        reqs = super().requires()

        reqs['unfiltered'] = data.ComputeObjectScaleVsHeightComposition(
            base_name=self.base_name,
            field_name="{}_flux".format(self.scalar),
            z_max=self.z_max,
            x="r_equiv",
            object_filters=None,
            mask_method=self.mask_method,
            mask_method_extra_args=self.mask_method_extra_args,
            object_splitting_scalar=self.object_splitting_scalar
        )

        return reqs

    def run(self):
        super().run()

        input = self.input()
        ds_filtered = self.input()['filtered_objects'].open()
        ds_unfiltered = self.input()['unfiltered'].open()

        name = "{}_flux__mean".format(self.scalar)

        def scale_flux(da_flux):
            if da_flux.sampling == 'full domain':
                return da_flux
            else:
                return da_flux*ds_filtered.areafrac.sel(sampling=da_flux.sampling)
        da_flux_tot = ds_filtered[name].groupby('sampling').apply(scale_flux)

        z_lims = self.z_pad, self.z_max-self.z_pad
        da_flux_tot_range = da_flux_tot.sel(
            zt=slice(*z_lims)
        )

        captured_fraction = (
            da_flux_tot_range.sel(sampling='objects')
            /da_flux_tot_range.sel(sampling='mask')
        ).mean(dim='zt')

        n_objects_total = ds_unfiltered.object_id.count().item()
        n_objects_filtered = ds_filtered.object_id.count().item()

        with open(self.output()['txt'].fn, 'w') as fh:
            fh.write("mean fraction of mask flux carried by objects"
                    " ({} / {} ~ {:.2f}%) between {}m and {}m: "
                    "{:.2f}%".format(
                n_objects_filtered, n_objects_total,
                100.*float(n_objects_filtered)/float(n_objects_total),
                *z_lims, 100.*captured_fraction.values
            ))

    def output(self):
        output = super().output()
        fn_txt = output['plot'].fn.replace('.png', '.txt')
        output['txt'] = luigi.LocalTarget(fn_txt)
        return output
