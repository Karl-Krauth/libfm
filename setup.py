import glob
import subprocess

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install

# This is a hack to add the compiled module to the
# data files after we run the makefile. This is because
# the name of the compiled file varies based on the platform
# it's compiled on.
data_files = []


class CustomInstallCommand(install):
    def run(self):
        subprocess.check_call(['make'])
        data_files.append(('pyfm', glob.glob('bin/pyfm*')))
        install.run(self)


class CustomDevelopCommand(develop):
    def run(self):
        subprocess.check_call(['make'])
        data_files.append(('pyfm', glob.glob('bin/pyfm*')))
        develop.run(self)


setup(
    name='pyfm',
    version='0.0.1',
    author='Karl Krauth',
    author_email='karl.krauth@gmail.com',
    description='A pybind11 wrapper for libfm.',
    packages=find_packages(),
    data_files=data_files,
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
    },
    install_requires=[
        'pybind11',
    ]
)
