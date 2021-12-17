# Changelog

## [v0.3.1](https://github.com/leifdenby/genesis/tree/v0.3.1)

[Full Changelog](https://github.com/leifdenby/genesis/v0.3.1...v0.3.0)

- clean up installation and continuous integration so that package can be
  installed from pypi as `cloud-genesis` [\#8](https://github.com/leifdenby/genesis/pull/8)


## [v0.3.0](https://github.com/leifdenby/genesis/tree/v0.3.0)

[Full Changelog](https://github.com/leifdenby/genesis/v0.3.0...v0.2.0)

**added features**:

- Extraction of 3D cloud-base state from 2D tracked clouds ([528b36](https://github.com/leifdenby/genesis/commit/5238b3696e4fde336e54e15783e7e657189f9014))

- Extraction of first rise for tracked 2D objects [9465b1a](https://github.com/leifdenby/genesis/commit/9465b1a182a64417fbabcd6dae256c7e6bdc965a)

- KDE-based density of flux decomposition plot [5508a12](https://github.com/leifdenby/genesis/commit/5508a12ef223cc78aadfab9c1e840d5a9d5c1b69)

- General function for 2D binned statistics of xr.DataArrays [f4768f7](https://github.com/leifdenby/genesis/commit/f4768f7c8537ce80632e21a0cdc4c4742cf5469a)

- Cumulant assymmetry on profile plots [8eeace9](https://github.com/leifdenby/genesis/commit/8eeace919965ccee8085525eef874c9980ab55ef)

- isometric rendering of idealised 3D outline shapes [03ae46c](https://github.com/leifdenby/genesis/commit/03ae46c67620ac74bb763c68faf130cca53ef62c)

- cumulants along only one dimension (e.g. for height profiles of horizontal
  cumulant) [fe4992d](https://github.com/leifdenby/genesis/commit/fe4992d8f3619aa151322473ade734a625f49b1e)

- offset 3D extraction from 2D tracking with Gallilean transform [1210fca](https://github.com/leifdenby/genesis/commit/1210fca44abbae3dd94a336815f0c851b9816b6a)

**bugfixes**:

- Fix annotation marker for cumulant anti-correlation [21a5715](https://github.com/leifdenby/genesis/commit/21a571522bfdebb541d4af17ea54e3288a63e4f6)

**maintenance**:

- Aggregation of properties for projected 2D objects refactored ([c3c8610](https://github.com/leifdenby/genesis/commit/c3c8610c0d504abea9a5aa79779f449a28177322))


## [v0.2.0](https://github.com/leifdenby/genesis/tree/v0.2.0)

First release to coincide with publication draft. Contains functionality for
identifying and characterising 3D coherent structures
