import subprocess

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        subprocess.check_call(['make'])


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        subprocess.check_call(['make'])


setup(
    name='pyfm',
    version='0.0.1',
    author='Karl Krauth',
    author_email='karl.krauth@gmail.com',
    description='A pybind11 wrapper for libfm.',
    packages=find_packages(),
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
    },
    install_requires=[
        'pybind11',
    ]
)
