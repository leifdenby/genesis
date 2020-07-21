"""
Simple utility script to run external cloud-tracking code from python
"""
import os
import signal
import subprocess
from enum import Enum


UCLALES_TRACKING_BIN_PATH = os.environ.get("UCLALES_TRACKING_BIN_PATH")

if UCLALES_TRACKING_BIN_PATH is None or not os.path.exists(UCLALES_TRACKING_BIN_PATH):
    HAS_TRACKING = False
else:
    HAS_TRACKING = True


def _execute(cmd):
    # https://stackoverflow.com/a/4417735
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()

    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


class TrackingType(Enum):
    CLOUD_CORE = "cloud,core"
    CLOUD_CORE_THERMAL = "cloud,core,thermal"
    THERMALS_ONLY = "thermal"

    @classmethod
    def make_identifier(cls, t):
        return t.value.replace(",", "_")


REQUIRED_FIELDS_MAPPING = dict(
    core=["core",],
    thermal=["lwp", "trcbase", "trctop", "trcpath"],
    cloud=["cldbase", "cldtop", "lwp"],
)


def get_required_fields(tracking_type):
    required_fields = [
        "core",
    ]
    for tracked_feature in tracking_type.value.split(","):
        required_fields += REQUIRED_FIELDS_MAPPING[tracked_feature]
    return set(required_fields)


def call(
    data_path,
    dataset_name,
    tn_start,
    tn_end,
    tracking_type=TrackingType.CLOUD_CORE,
    U_offset=None,
):
    """
    Call the UCLALES cloud tracking library using data in `data_path` with
    `dataset_name` in time-step interval `tn_start`:`tn_end` tracking only the
    objects of `tracking_type`
    """
    if not HAS_TRACKING:
        raise Exception(
            """Couldn't find external cloud-tracking utility,
        please set the UCLALES_TRACKING_BIN_PATH environment variable"""
        )

    old_dir = os.getcwd()

    try:
        os.chdir(data_path)
        args = [
            UCLALES_TRACKING_BIN_PATH,
            dataset_name,
            str(tn_start),
            str(tn_end),
            tracking_type.value,
        ]

        if U_offset is not None:
            args += U_offset

        print(data_path)
        print((" ".join(args)))

        for output in _execute(args):
            print((output.strip()))

    except subprocess.CalledProcessError as ex:
        return_code = ex.returncode
        error_extra = ""
        if -return_code == signal.SIGSEGV:
            error_extra = ", the utility segfaulted "

        raise Exception(
            "There was a problem when calling the tracking "
            "utility (errno={}): {} {}".format(error_extra, return_code, ex)
        )
    finally:
        os.chdir(old_dir)

    return os.path.join(data_path, "{}.out.xy.track.nc".format(dataset_name))
