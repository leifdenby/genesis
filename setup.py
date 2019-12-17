from setuptools import setup, find_packages

# note: version is maintained inside convml_tt/version.py
exec(open('genesis/version.py').read())

setup(
    name='genesis',
    packages=find_packages(exclude=['contrib', 'tests', 'docs']),
    version=__version__,
    description='GENESIS toolkit for analysing coherent structures in atmospheric flows',
    author='Leif Denby',
    author_email='leifdenby@gmail.com',
    url='https://github.com/leifdenby/genesis',
    classifiers=[],
)
