# interpol/test/test-4d.py 11mar
# http://stackoverflow.com/questions/14119892/python-4d-linear-interpolation-on-a-rectangular-grid


import sys
import numpy as np
import scipy

from intergrid import Intergrid
import util as ut

print(
    "versions: numpy %s  scipy %s  python %s"
    % (np.__version__, scipy.__version__, sys.version.split()[0])
)

dim = 5  # > 4: add shape 1
scale = 3  # 361 * 720 * 47 * 8 = 98M  / scale^3
dtype = np.float64
nquery = 10000
order = 1
seed = 0

# run this.py scale= ... from sh or ipython --
exec("\n".join(sys.argv[1:]))
np.set_printoptions(2, threshold=100, edgeitems=10, suppress=True)
np.random.seed(seed)
nquery = int(nquery)

# ...............................................................................
lats = np.arange(-90, 90.5, 0.5 * scale)
lons = np.arange(-180, 180, 0.5 * scale)
alts = np.arange(1, 1000, 21.717 * scale)
times = np.arange(8)
shape = (len(lats), len(lons), len(alts), len(times))
if dim > 4:
    shape += (dim - 4) * (1,)
ngrid = np.prod(shape)
lo = np.r_[lats[0], lons[0], alts[0], times[0], (dim - 4) * [0]]
hi = np.r_[lats[-1], lons[-1], alts[-1], times[-1], (dim - 4) * [1]]

griddata = np.arange(ngrid, dtype=dtype).reshape(shape)
# order=1 must be exact for linear
query_points = lo + np.random.uniform(size=(nquery, dim)).astype(dtype) * (hi - lo)
print(
    "shape %s  %.2gM * %d  nquery %d" % (shape, ngrid / 1e6, griddata.itemsize, nquery)
)

linmaps = []
# print "linmaps:"
for l, h, n in zip(lo, hi, shape):
    linmap = np.linspace(l, h, n)
    linmaps += [linmap]
    # print linmap

# ...............................................................................
out = []
for maps in [[], linmaps]:
    intergrid = Intergrid(griddata, lo=lo, hi=hi, maps=maps, order=order, verbose=1)
    query_out = intergrid.at(query_points)
    out.append(query_out)
    # print "intergrid at lo %s: = %.3g" % (lo, intergrid(lo))
    # print "intergrid at hi %s: = %.3g" % (hi, intergrid(hi))

print("|no maps - pwl maps|:", end=" ")
ut.avmaxdiff(out[0], out[1], query_points)

# versions: numpy 1.6.2  scipy 0.11.0  python 2.7.3  2.5 GHz iMac
# scale 1: shape (361, 720, 47, 8)  98M * 8
# Intergrid: 617 msec  1000000 points in a (361, 720, 47, 8) grid  0 maps  order 1
# Intergrid: 788 msec  1000000 points in a (361, 720, 47, 8) grid  4 maps  order 1
# max diff no maps / pwl maps: 7.5e-08 = 9.29e+07 - 9e+07  at [  81.06  151.58  947.47    4.6 ]
