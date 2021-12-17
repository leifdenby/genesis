import itertools

import matplotlib.patches as mpatches
import matplotlib.pyplot as plot
import numpy as np
import scipy.spatial.distance
import skimage
import skimage.color
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation
from scipy.signal import convolve2d


def _expand_int_regions(v, s):
    # define the convolution kernel
    k = np.ones((s * 2 + 1, s * 2 + 1))

    # attempt at circular kernel
    i = np.arange(-s, s + 1)
    d = np.linalg.norm(np.meshgrid(i, i), axis=0)
    k = 0.5 * (np.tanh((s - d)) + 1.0)

    # do the convolution to get contributions from neighbouring cells
    l = convolve2d(v, k, mode="same")  # noqa
    # find out how many cells contributed to each one that has been newly
    # filled
    l_s = convolve2d(v != 0, k, mode="same")
    # make sure don't devide by zero
    l_s[l_s == 0] = 1

    v_expanded = l / l_s

    v_expanded = np.ma.masked_array(v_expanded, v_expanded == 0)

    # fill with zeros, gives us an extra unique value
    # if not np.all(np.unique(v.filled(0.)) ==
    # np.unique(v_expanded.filled(0.))):
    #    raise Exception("Some expanded regions overlap")

    return v_expanded


def make_cloud_surroundings_mask(cloud_mask, s):
    cloud_mask_expanded = _expand_int_regions(cloud_mask, s)
    cloud_mask_expanded = np.ma.masked_array(cloud_mask_expanded, cloud_mask != 0)

    return cloud_mask_expanded


def get_cloudmask_at_height(t, z_slice, cloud_data, do_plot=False):
    rl_slice = cloud_data.get_from_3d(var_name="l", z=z_slice, t=t, debug=False)

    # cutoff in UCLALES for `cldbase` is rl=0.0kg/kg, so we use the same here
    image = rl_slice > 0.0

    # apply threshold
    thresh = skimage.filters.threshold_otsu(image)
    bw = skimage.morphology.closing(image > thresh, skimage.morphology.square(3))

    # remove artifacts connected to image border
    cleared = bw.copy()
    skimage.segmentation.clear_border(cleared)

    # label image regions
    label_image = skimage.measure.label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1

    if do_plot:
        image_label_overlay = skimage.color.label2rgb(label_image, image=image)
        fig, ax = plot.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(image_label_overlay)

        for region in skimage.measure.regionprops(label_image):
            # draw rectangle around segmented regions
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(rect)

        plot.xlim(0, 500)
        plot.ylim(0, 500)
        plot.show()

    return label_image


def get_cloud_surroundings_at_height(cloud_set, tn, z_slice, var_name):
    cloud_data = cloud_set.cloud_data
    t = tn * cloud_data.dt

    # load the data up already, so that we're sure it exists
    data_slice = cloud_data.get_from_3d(var_name=var_name, z=z_slice, t=t, debug=False)

    # only look at environment of clouds which are high enough to be in the
    # slice at `z_slice`
    cloud_set__tall_enough = cloud_set.filter(cloudtop_height__gt=z_slice, _tn=tn)

    print(
        (
            "Getting near-environment data for {} clouds".format(
                len(cloud_set__tall_enough)
            )
        )
    )

    cloudbase_center_cloudset = cloud_set__tall_enough.get_value(
        "cloudbase_center_position", tn=tn
    )
    # remove all the clouds where the cloud-base is poorly defined
    cloudbase_center_cloudset = cloudbase_center_cloudset[
        ~np.isnan(cloudbase_center_cloudset[:, 0])
    ]
    cloudbase_center_cloudset.shape

    cloudmask_slice = get_cloudmask_at_height(
        t=t, z_slice=z_slice, cloud_data=cloud_data
    )
    labelled_regions_center = cloud_data.dx * np.array(
        [reg.centroid for reg in skimage.measure.regionprops(cloudmask_slice)]
    )

    # for each cloud in the cloud_set find the nearest region in the slice
    # above, we will assume this is part of the same cloud
    d = scipy.spatial.distance.cdist(cloudbase_center_cloudset, labelled_regions_center)
    min_dist_label_num = np.argmin(d, axis=1)

    # remove all the labels not in `min_dist_label_num`, i.e. only keep a
    # region in the slice for each cloud we have
    cloudmask_slice_filtered = np.zeros_like(cloudmask_slice)
    for l in np.unique(cloudmask_slice):
        if l in min_dist_label_num:
            cloudmask_slice_filtered[cloudmask_slice == l] = l

    m = cloudmask_slice_filtered
    m_current = np.array(m).astype(int)

    yield (0, data_slice[m_current != 0], data_slice, m_current)

    for n in itertools.count():
        m_ring = make_cloud_surroundings_mask(m_current, s=1).astype(int)
        data_ring = data_slice[m_ring != 0]

        m_current += m_ring

        yield (cloud_data.dx * (n + 1), data_ring, data_slice, m_current)


def get_cloud_surroundings(cloud_mask, da_scalar):
    pass
