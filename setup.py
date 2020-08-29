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


def _check_bazel():
    """Check Bazel binary as well as its version."""

    if not spawn.find_executable('bazel'):
        sys.stderr.write('could not find bazel executable. Please install bazel to'
                         'build the MediaPipe Python package.')
        sys.exit(-1)
    try:
        bazel_version_info = subprocess.check_output(['bazel', '--version'])
    except subprocess.CalledProcessError:
        sys.stderr.write('fail to get bazel version by $ bazel --version.')
    bazel_version_info = bazel_version_info.decode('UTF-8').strip()
    version = bazel_version_info.split('bazel ')[1].split('-')[0]
    version_segments = version.split('.')
    # Treat "0.24" as "0.24.0"
    if len(version_segments) == 2:
        version_segments.append('0')
    for seg in version_segments:
        if not seg.isdigit():
            sys.stderr.write('invalid bazel version number: %s\n' %
                             version_segments)
            sys.exit(-1)
    bazel_version = int(''.join(['%03d' % int(seg)
                                 for seg in version_segments]))
    if bazel_version < 2000000:
        sys.stderr.write(
            'the current bazel version is older than the minimum version that MediaPipe can support. Please upgrade bazel.'
        )


class BazelExtension(setuptools.Extension):
    """A C/C++ extension that is defined as a Bazel BUILD target."""

    def __init__(self, bazel_target, target_name=''):
        self.bazel_target = bazel_target
        self.relpath, self.target_name = (
            posixpath.relpath(bazel_target, '//').split(':'))
        if target_name:
            self.target_name = target_name
        ext_name = os.path.join(
            self.relpath.replace(posixpath.sep, os.path.sep), self.target_name)
        setuptools.Extension.__init__(self, ext_name, sources=[])


class BuildBazelExtension(build_ext.build_ext):
    """A command that runs Bazel to build a C/C++ extension."""

    def run(self):
        _check_bazel()
        for ext in self.extensions:
            self.bazel_build(ext)
        build_ext.build_ext.run(self)

    def bazel_build(self, ext):
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        bazel_argv = [
            'bazel',
            'build',
            '--compilation_mode=opt',
            '--copt', '-DMESA_EGL_NO_X11_HEADERS',
            '--copt', '-DEGL_NO_X11',
            str(ext.bazel_target + '.so'),
        ]
        self.spawn(bazel_argv)
        ext_bazel_bin_path = os.path.join('bazel-bin', ext.relpath,
                                          ext.target_name + '.so')
        ext_dest_path = self.get_ext_fullpath(ext.name)
        ext_dest_dir = os.path.dirname(ext_dest_path)
        if not os.path.exists(ext_dest_dir):
            os.makedirs(ext_dest_dir)
        # print(ext_bazel_bin_path)
        # print(ext_dest_path)
        # ext_dest_path = os.path.join(
        #     os.path.dirname(os.path.abspath(__file__)), ext.target_name + '.so')
        shutil.copyfile(ext_bazel_bin_path, ext_dest_path)


setuptools.setup(
    name='pikapi',
    version=__version__,
    url='https://github.com/xiong-jie-y/pikapi',
    description='Pikapi is a perception library for interaction with agents.',
    author='xiong jie',
    author_email='fwvillage@gmail.com',
    long_description=open(os.path.join(MP_ROOT_PATH, 'README.md')).read(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['apps.*', 'development.*']),
    install_requires=_parse_requirements('requirements.txt'),
    cmdclass={
        'build_ext': BuildBazelExtension,
    },
    ext_modules=[
        BazelExtension('//pikapi:graph_runner'),
    ],
    zip_safe=False,
    include_package_data=True,
    classifiers=[
    ],
    license='Apache 2.0',
    keywords='perception',
)
