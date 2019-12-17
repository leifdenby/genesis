# GENESIS toolkit for analysing coherent structures in atmospheric flows

## Installation

All the necessary dependencies can be installed with
[conda](https://www.anaconda.com/distribution/). Once conda is installed we can
create a conda environment and install the dependecies into there:


```
conda create -n genesis -f environment.yml
conda activate genesis
```

Finally you will need to install the object identification code which is in
a seperate repository

```bash
pip install git+git://github.com/leifdenby/cloud_identification\#egg\=cloud-identification
```
