from setuptools import setup, find_packages


requirements = [
    'h5py',
    'pyyaml',
    'tqdm',
    'numpy',
    'torch',
    'torch3d'
]

__version__ = '0.1.0'

setup(
    name='sota3d',
    version=__version__,
    author='Quang-Hieu Pham',
    author_email='pqhieu1192@gmail.com',
    url='https://github.com/pqhieu/sota3d',
    install_requires=requirements,
    packages=find_packages()
)
