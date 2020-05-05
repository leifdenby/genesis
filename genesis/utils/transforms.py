
def _wrap_add(x, y, a, b):
    """add x+y modulu the answer being in range [a...b[
    
    https://stackoverflow.com/a/51467186/271776
    """
    y %= b - a
    x = x + y
    return x - (b-a)*(x >= b)


def offset_gal_single(da_coord, U, tref):
    """
    Remove Galilean transform offset with (x, y)-velocity vector `U` and
    reference time `tref`.
    """

    if da_coord.name.startswith('x'):
        n = 0
    elif da_coord.name.startswith('y'):
        n = 1
    else:
        raise NotImplementedError(da_coord.name)

    dt = (da_coord.time - tref).dt.seconds.item()
    da_ = da_coord.copy()

    return _wrap_add(da_, dt*U[n], da_.min(), da_.max())


def offset_gal(da, U, tref, truncate_to_grid=False):
    """
    Remove Galilean transform offset with (x, y)-velocity vector `U` and
    reference time `tref`. If `truncate_to_grid` nearest position in original
    grid will be returned, otherwise grid locations will be purely based on
    transform
    """
    dt = (da.time - tref).dt.seconds.item()

    da_ = da.copy()
    x_ = da_.xt.data
    y_ = da_.yt.data
    
    da_['x_offset'] = ('xt',), _wrap_add(x_, dt*U[0], x_.min(), x_.max())
    da_['y_offset'] = ('yt',), _wrap_add(y_, dt*U[1], y_.min(), y_.max())

    da_ = da_.swap_dims(dict(xt='x_offset', yt='y_offset'))
    da_ = da_.drop(['xt', 'yt'])
    da_ = da_.sortby(['x_offset', 'y_offset'])

    if truncate_to_grid:
        da_['xt'] = ('x_offset', ), da.xt.values
        da_['yt'] = ('y_offset', ), da.yt.values
        da_ = da_.swap_dims(dict(x_offset='xt', y_offset='yt'))
        da_ = da_.drop(['x_offset', 'y_offset'])
        return da_
    else:
        da_ = da_.rename(dict(x_offset='xt', y_offset='yt'))
        return da_
