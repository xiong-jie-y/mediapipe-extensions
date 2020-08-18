from distutils import spawn
import distutils.command.build as build
import distutils.command.clean as clean
import glob
import os
import posixpath
import shutil
import subprocess
import sys

import setuptools
import setuptools.command.build_ext as build_ext
import setuptools.command.install as install

__version__ = '0.79'
MP_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
# ROOT_INIT_PY = os.path.join(MP_ROOT_PATH, '__init__.py')
# if not os.path.exists(ROOT_INIT_PY):
#   open(ROOT_INIT_PY, 'w').close()

def _parse_requirements(path):
  with open(os.path.join(MP_ROOT_PATH, path)) as f:
    return [
        line.rstrip()
        for line in f
        if not (line.isspace() or line.startswith('#'))
    ]


setuptools.setup(
    name='pika',
    version=__version__,
    url='https://github.com/xiong-jie-y/pika',
    description='Pika is a perception library for interaction with agents.',
    author='xiong jie',
    author_email='fwvillage@gmail.com',
    long_description=open(os.path.join(MP_ROOT_PATH, 'README.md')).read(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['apps.*', 'development.*']),
    install_requires=_parse_requirements('requirements.txt'),
    cmdclass={
    },
    ext_modules=[
    ],
    zip_safe=False,
    include_package_data=True,
    classifiers=[
    ],
    license='Apache 2.0',
    keywords='perception',
)
