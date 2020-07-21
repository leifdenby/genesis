# test-2d.py: interpolate data on a productgrid x, y -> another productgrid xnew, ynew
# 2013-12-04 dec


import sys
import numpy as np
from scipy import __version__

import intergrid  # http://denis-bz.github.com/docs/intergrid.html
# from grids import productgrid, func_at

def productgrid( *xyz ):
    """ productgrid( 1d arrays x, y, z )
        -> [[x0 y0 z0] [x0 y0 z1] ... [xlast ylast zlast]]
    """
        # meshgrid mgrid ogrid ? mttiw
    from itertools import product
    return np.array( list( product( *xyz )), dtype=np.float32 )

def func_at( func, points, **kw ):
    return np.array([ func( p, **kw ) for p in points ])  # fromiter 1d only

def avmax( x ):
    return "av %.2g  max %.2g" % (x.mean(), x.max())

#...............................................................................
Nx, Ny = 100, 100
Nxnew, Nynew = 100, 100
ncycle = 5  # * sqrt(dim) along diagonal
order = 1  # 1 bilinear, 3 spline
prefilter = 0  # 0: smoothing B-spline, 1/3: M-N, 1: interpolating 
plot = 0
seed = 0

    # run this.py a=1 b=None c=[3] 'd = expr' ...  from sh or ipython
exec( "\n".join( sys.argv[1:] ))
np.set_printoptions( 1, threshold=100, edgeitems=10, suppress=True )
np.random.seed(seed)

print("versions: numpy %s  scipy %s  python %s  maxint %x" % (
    np.__version__, __version__, sys.version.split()[0], sys.maxsize ))

title = "intergrid:  Nx %d Ny %d  Nxnew %d Nynew %d  ncycle %d " % (
    Nx, Ny, Nxnew, Nynew, ncycle )
print(80 * "-")
print(title)

    # data grid: a random box grid (error >> smoothly varying) --
x = np.sort( np.r_[ 0, np.random.random( Nx - 2 ), 1 ])
y = np.sort( np.r_[ 0, np.random.random( Ny - 2 ), 1 ])
datagrid = productgrid( x, y )

    # new grid: uniform --
xnew = np.linspace( 0, 1, Nxnew )
ynew = np.linspace( 0, 1, Nynew )
newgrid = productgrid( xnew, ynew )  # shape (Nxnew * Nynew, 2)

#...............................................................................
def func( X ):  # ncycle
    r = np.sqrt( np.sum( X**2, axis=-1 ))
    return np.sin( 2 * np.pi * ncycle * r ) * 100

    # exact func on new grid, to compare with interpolated --
func_exact = func_at( func, newgrid ) .reshape(Nxnew, Nynew)
func0 = func_exact[:20,0]
print("func[:20,0]", func0 .round().astype(int), "\n")

#...............................................................................
Data = func_at( func, datagrid ).astype(np.float32)  # 1d, Nx * Ny

interfunc = intergrid.Intergrid(
    Data.reshape( Nx, Ny ),
    lo=0, hi=1,     # corners of data grid
    maps=[x, y],    # Data grid
    order=order, prefilter=prefilter, verbose=1 )

interpolateddata = interfunc( newgrid ) .reshape(Nxnew, Nynew)

diff = interpolateddata - func_exact
print("order %d  prefilter %.2g  %s  |f - intergrid|: %s \n" % (
    order, prefilter, interpolateddata.dtype, avmax( np.fabs(diff) )))

if plot:
    import pylab as pl
    pl.title( title )
    imshow = pl.imshow( diff, cmap=pl.cm.Spectral_r )
        # rugs: big gaps in x, y => large error
    pl.plot( x * Nx, np.zeros(Nx), marker="|", markersize=10, linestyle="none", antialiased=False )
    pl.plot( np.zeros(Ny), y * Ny, marker="_", markersize=10, linestyle="none", antialiased=False )
    pl.gcf().colorbar( imshow )
    pl.show()

