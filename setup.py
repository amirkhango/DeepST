from setuptools import setup
from setuptools import find_packages


setup(name='DeepST',
      version='0.0.1',
      description='Deep Learning for Spatio-Temporal Data',
      author='Junbo Zhang',
      author_email='zjb2046@gmail.com',
      url='https://github.com/lucktroy/DeepST',
      download_url='https://github.com/lucktroy/DeepST/',
      license='MIT',
      install_requires=['keras', 'theano'],
      extras_require={
          'h5py': ['h5py'],
          'visualize': ['pydot-ng'],
      },
      packages=find_packages())