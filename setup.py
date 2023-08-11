"""SeismicPro is a library for seismic data processing."""

import re
from setuptools import setup, find_packages


with open('./seismicpro/__init__.py', 'r', encoding='utf-8') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

with open('./README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='SeismicPro',
    packages=find_packages(exclude=['tutorials', 'benchmark']),
    version=version,
    url='https://github.com/gazprom-neft/SeismicPro',
    license='Apache License 2.0',
    author='Gazprom Neft DS team',
    author_email='rhudor@gmail.com',
    description='A framework for seismic data processing',
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    platforms='any',
    include_package_data=True,
    install_requires=[
        'numpy>=1.20',
        'scipy>=1.7',
        'numba>=0.57',
        'pandas>=1.3',
        'polars[pyarrow]>=0.18.7',
        'scikit-learn>=0.23.2',
        'opencv_python>=4.5.1',
        'rustworkx>=0.12.1',
        'segyio>=1.9.5',
        'segfast>=1.0.1',
        'tqdm>=4.56.0',
        'pytest>=6.0.1',
        'torch>=1.8',
        'matplotlib>=3.5.1',
        'seaborn>=0.11.1',
        'dill>=0.3.3',
        'multiprocess>=0.70.11',
        'requests>=2.24',
        'psutil>=5.7.2',
        'batchflow>=0.8.7',
        'tbb>=2021.7.1',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
    ],
)
