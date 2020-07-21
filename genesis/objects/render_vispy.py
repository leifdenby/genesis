# -*- coding: utf-8 -*-

"""
Controls:

* 1  - toggle camera between first person (fly), regular 3D (turntable) and
       arcball
* 2  - toggle between volume rendering methods
* 3  - toggle between stent-CT / brain-MRI image
* 4  - toggle between colormaps
* 0  - reset cameras
* [] - decrease/increase isosurface threshold

With fly camera:

* WASD or arrow keys - move around
* SPACE - brake
* FC - move up-down
* IJKL or mouse - look around

Axes are x=red, y=green, z=blue.
"""

from itertools import cycle
import os

import xarray as xr

from vispy import app, scene
from vispy.color import get_colormaps, BaseColormap


# create colormaps that work well for translucent and additive volume rendering
class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.2);
    }
    """


class TransGrays2(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(0.5, t, t*t, max(0, t*t-0.2));
    }
    """


class TransGrays3(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(0.5, t, t, max(0, t-0.2));
    }
    """


# Setup colormap iterators
opaque_cmaps = cycle(get_colormaps())
translucent_cmaps = cycle([TransGrays2(), TransGrays3(), TransGrays(), TransFire()])
opaque_cmap = next(opaque_cmaps)
translucent_cmap = next(translucent_cmaps)


def main(objects, object_id, var_name=None):
    m = objects.where(objects == object_id, drop=True)

    data_source, objects_mask = object_file.split(".objects.")
    path, base_name = data_source.split("__")
    if var_name:
        fn = os.path.join(
            path, "3d_blocks", "full_domain", "{}.{}.nc".format(base_name, var_name)
        )
        if not os.path.exists(fn):
            fn = os.path.join(path, "masks", "{}.{}.nc".format(base_name, var_name))
        da = xr.open_dataarray(fn, decode_times=False)
        da = da.squeeze().sel(xt=m.xt, yt=m.yt, zt=m.zt)

        da -= da.mean()
        da /= da.max()
        print(da.max())
    else:
        da = m

    # vispy expects data as (z, y, x)
    da = da.transpose("zt", "yt", "xt")
    nz, ny, nx = da.shape

    fn_out = "temp_3d_block.nc"
    del da.xt.attrs["standard_name"]
    del da.yt.attrs["standard_name"]
    da.to_netcdf(fn_out)
    print("Wrote {}".format(fn_out))

    # m = np.swapaxes(m, 0, 2)

    # x_min, x_max = m.xt.min(), m.xt.max()
    # y_min, y_max = m.yt.min(), m.yt.max()
    # xc, yc = m.xt.mean(), m.yt.mean()

    # Prepare canvas
    canvas = scene.SceneCanvas(keys="interactive", size=(1200, 800), show=True)
    canvas.measure_fps()

    # Set up a viewbox to display the image with interactive pan/zoom
    view = canvas.central_widget.add_view()

    # Set whether we are emulating a 3D texture
    emulate_texture = False

    # Create the volume visuals, only one is visible
    # data = np.swapaxes(m.values, 0, 1)
    data = da.fillna(0.0).values  # np.flipud(np.rollaxis(m.values, 1))

    data = (data - data.min()) / (data.max() - data.min())

    volume1 = scene.visuals.Volume(
        data,
        parent=view.scene,
        threshold=0.5,
        emulate_texture=emulate_texture,
        method="translucent",
        cmap=translucent_cmap,
    )
    # volume1.transform = scene.STTransform(translate=(0, 0, nz))
    volume1.transform = scene.STTransform(translate=(-nx / 2, -ny / 2, 0))

    # Create three cameras (Fly, Turntable and Arcball)
    fov = 60.0
    cam = scene.cameras.ArcballCamera(
        parent=view.scene, fov=fov, name="Arcball", up="z"
    )
    cam.distance = nz * 5
    view.camera = cam

    _ = scene.Axis(
        pos=[[-0.5 * nx, -0.5 * ny], [0.5 * nx, -0.5 * ny]],
        tick_direction=(0, -1),
        font_size=16,
        axis_color="r",
        tick_color="r",
        text_color="r",
        parent=view.scene,
    )
    _ = scene.Axis(
        pos=[[-0.5 * nx, -0.5 * ny], [0.5 * nx, -0.5 * ny]],
        tick_direction=(0, -1),
        font_size=16,
        axis_color="r",
        tick_color="r",
        text_color="r",
        parent=view.scene,
    )
    # xax.transform = scene.STTransform(translate=(0, 0, -0.2))

    yax = scene.Axis(
        pos=[[-0.5 * nx, -0.5 * ny], [-0.5 * nx, 0.5 * ny]],
        tick_direction=(-1, 0),
        font_size=16,
        axis_color="b",
        tick_color="b",
        text_color="b",
        parent=view.scene,
    )
    yax.transform = scene.STTransform(translate=(0, 0, -0.2))

    # Implement key presses
    @canvas.events.key_press.connect
    def on_key_press(event):
        global opaque_cmap, translucent_cmap
        if event.text == "1":
            pass
        elif event.text == "2":
            methods = ["translucent", "additive"]
            method = methods[(methods.index(volume1.method) + 1) % len(methods)]
            print("Volume render method: %s" % method)
            cmap = opaque_cmap if method in ["mip", "iso"] else translucent_cmap
            volume1.method = method
            volume1.cmap = cmap
        elif event.text == "4":
            if volume1.method in ["mip", "iso"]:
                cmap = opaque_cmap = next(opaque_cmaps)
            else:
                cmap = translucent_cmap = next(translucent_cmaps)
            volume1.cmap = cmap
        elif event.text == "0":
            cam.set_range()
        elif event.text != "" and event.text in "[]":
            s = -0.025 if event.text == "[" else 0.025
            volume1.threshold += s
            th = volume1.threshold
            print("Isosurface threshold: %0.3f" % th)

    # for testing performance
    # @canvas.connect
    # def on_draw(ev):
    # canvas.update()

    app.run()


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument("object_file", type=str)
    argparser.add_argument("--object-id", type=int, required=True)
    argparser.add_argument("--var-name", type=str)

    args = argparser.parse_args()

    object_file = args.object_file.replace(".nc", "")

    if "objects" not in object_file:
        raise Exception()

    base_name, objects_mask = object_file.split(".objects.")

    fn_objects = "{}.nc".format(object_file)
    if not os.path.exists(fn_objects):
        raise Exception("Couldn't find objects file `{}`".format(fn_objects))
    objects = xr.open_dataarray(fn_objects, decode_times=False)

    if args.object_id not in objects.values:
        raise Exception()

    print(__doc__)
    main(objects=objects, object_id=args.object_id, var_name=args.var_name)
