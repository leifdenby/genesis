import luigi


from .....objects.models import parcel_rise


class ParcelRiseModelFit(luigi.Task):
    base_name = luigi.Parameter()
    label_var = "cloud"
    var_name = "cloudtop"
    op = luigi.Parameter()
    dx = 25.0
    use_relative_time_axis = luigi.BoolParameter(default=True)

    track_without_gal_transform = luigi.BoolParameter(default=False)
    tracking_type = luigi.EnumParameter(enum=TrackingType)
    tracking_timestep_interval = luigi.ListParameter([])

    def requires(self):
        common_kws = dict(
            base_name=self.base_name,
            tracking_type=TrackingType.CLOUD_CORE,
            track_without_gal_transform=self.track_without_gal_transform,
            tracking_timestep_interval=self.tracking_timestep_interval,
        )

        return dict(
            z=AllObjectsAll2DCrossSectionAggregations(
                label_var=self.label_var,
                var_name=self.var_name,
                op="histogram",
                dx=self.dx,
                **common_kws,
            ),
            cloudtype=TrackingVariable2D(
                var_name="smcloudtype",
                **common_kws,
            )
        )


    def run(self):
        input = self.inpput()
        da_z = input['z'].open()
        da_cloudtype = input['cloudtype'].open()

        da.attrs['long_name'] = "number of cells"
        da.attrs['units'] = "1"

        da_obj = da_.sel(object_id=object_id)

        da_cloudtype = (
            da_cloudtype
            .rename(smcloudid="object_id")
            .astype(int)
            .drop("smcloud")
        )
        da_cloudtype['object_id'] = da_cloudtype.coords['object_id'].astype(int)

        da_single_cloud = da_cloudtype.where(da_cloudtype == 2, drop=True)
        da_single_cloud

        da_ = da.sel(object_id=da_single_cloud.object_id)

        ds_model_summary = parcel_rise.fit_model_and_summarise(z=z, t=t, predictions="mean_with_quantiles")

    def output(self):
        type_id = uclales_2d_tracking.TrackingType.make_identifier(self.tracking_type)
        if self.tracking_timestep_interval:
            interval_id = "tn{}_to_tn{}".format(*self.tracking_timestep_interval)
        else:
            interval_id = "__all__"

        name_parts = [
            self.var_name,
            f"of_{self.label_var}",
            f"tracked_{type_id}",
            self.op + ["", f"__{str(self.dx)}"][self.dx != None],
            interval_id,
        ]

        if not self.use_relative_time_axis:
            name_parts.append("absolute_time")

        if self.timestep_skip is not None:
            name_parts.append(f"{self.timestep_skip}tn_skip")

        if self.track_without_gal_transform:
            name_parts.append("go_track")

        fn = f"{'.'.join(name_parts)}.nc"
        p = get_workdir() / self.base_name / "cross_sections" / "aggregated" / fn
        return XArrayTarget(str(p))

