import subprocess

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install


class CustomInstallCommand(install):
    def run(self):
        subprocess.check_call(['make'])
        super().run(self)


class CustomDevelopCommand(develop):
    def run(self):
        subprocess.check_call(['make'])
        super().run(self)


class CustomEggInfoCommand(egg_info):
    def run(self):
        subprocess.check_call(['make'])
        super().run(self)


setup(
    name='pyfm',
    version='0.0.1',
    author='Karl Krauth',
    author_email='karl.krauth@gmail.com',
    description='A pybind11 wrapper for libfm.',
    long_description='',
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
        'egg_info': CustomEggInfoCommand,
    },
    zip_safe=False,
)
