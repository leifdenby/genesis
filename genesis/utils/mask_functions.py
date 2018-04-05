import numpy as np


def w_pos(w_zt):
    return w_zt > 0.0
w_pos.description = "positive cell-centered vertical velocity"


def coldpool_coarse(tv0100, w_zt):
    return tv0100 < -0.1
coldpool_coarse.description = 'coarse coldpool detection using -0.1K from mean of theta_v at 100m'
