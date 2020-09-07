import networkx as nx
import numba
import numpy as np
import xarray as xr


@numba.njit
def map_labels(labels_orig, mapping):
    labels = labels_orig.copy()
    nt, nx, ny = labels.shape  # noqa
    for tn in np.arange(nt):
        for i in np.arange(nx):
            for j in np.arange(ny):
                labels[tn, i, j] = mapping[labels[tn, i, j]]
    return labels


def create_tracking_family_2D_field(ds_tracking):
    ds_thermals = xr.merge(
        [
            ds_tracking.smthrmtop.rename("z_top"),
            ds_tracking.smthrmtmin.rename("t_min"),
            ds_tracking.smthrmchild.rename("cloud_ids"),
            ds_tracking.smthrmtmax.rename("t_max"),
        ]
    )
    ds_thermals = ds_thermals.rename(dict(smthrmid="thermal_id", smthrmt="t_thermal"))
    # ds_thermals = ds_thermals.expand_dims('has_cloud')
    ds_thermals["has_cloud"] = np.logical_not(
        ds_thermals.cloud_ids.isel(thrmrel=0).isnull()
    ).compute()

    # select thermals which are connected to clouds
    thermal_ids_with_cloud = ds_thermals.where(
        ds_thermals.has_cloud, drop=True
    ).thermal_id
    ds_thermals_wc = ds_thermals.sel(thermal_id=thermal_ids_with_cloud)

    def _build_thermal_cloud_graph(ds_):
        g = nx.Graph()
        for thermal_id in ds_.thermal_id.values:
            ds_thermal = ds_.sel(thermal_id=thermal_id)
            cloud_ids = ds_thermal.cloud_ids.dropna(dim="thrmrel")
            for cloud_id in cloud_ids.values:
                thermal_node = f"t_{int(thermal_id)}"
                cloud_node = f"c_{int(cloud_id)}"
                if not thermal_node in g:
                    g.add_node(thermal_node, color="red", label=thermal_node)
                if not cloud_node in g:
                    g.add_node(cloud_node, color="blue")
                g.add_edge(thermal_node, cloud_node)
        return g

    g = _build_thermal_cloud_graph(ds_thermals_wc)

    # build up collection of single-connection thermal <--> cloud
    single_connection = []
    for g_sub in nx.algorithms.components.connected_components(g):
        if len(g_sub) == 2:
            single_connection.append(g_sub)

    # creating mapping we can use later for deciding the labels to use
    cloud_family_map = np.zeros(int(ds_tracking.smcloudid.max()))
    thermal_family_map = np.zeros(int(ds_tracking.smthrmid.max()))

    # check that there isn't an overflow in the index to family mapping so that
    # we can't seperate thermal and cloud id "{thermal_id}00{cloud_id}" is what
    # we get
    thermal_id_max = 1000000
    assert int(ds_tracking.smthrmid.max()) < thermal_id_max

    for labels in single_connection:
        c_label, t_label = sorted(labels)
        c_id = int(c_label.replace("c_", ""))
        t_id = int(t_label.replace("t_", ""))
        family_id = t_id + (thermal_id_max * 100) * c_id
        cloud_family_map[c_id] = family_id
        thermal_family_map[t_id] = family_id

    da_labels_orig = ds_tracking.nrcloud.fillna(0)
    labels_orig = da_labels_orig.values.astype(int)

    labels_mapped = map_labels(labels_orig, mapping=cloud_family_map)

    da_labels_mapped = xr.DataArray(
        labels_mapped,
        coords=da_labels_orig.coords,
        dims=da_labels_orig.dims,
        attrs=da_labels_orig.attrs,
    )

    return da_labels_mapped
