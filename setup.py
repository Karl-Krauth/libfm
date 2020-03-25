import subprocess

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install


class CustomInstallCommand(install):
    def run(self):
        subprocess.check_call(['make'])
        install.run(self)


class CustomDevelopCommand(develop):
    def run(self):
        subprocess.check_call(['make'])
        develop.run(self)


setup(
    name='pyfm',
    version='0.0.1',
    author='Karl Krauth',
    author_email='karl.krauth@gmail.com',
    description='A pybind11 wrapper for libfm.',
    packages=find_packages(),
    data_files=['pyfm', ['bin/pyfm*']],
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
    },
    install_requires=[
        'pybind11',
    ]
)
