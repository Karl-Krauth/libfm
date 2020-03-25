import glob
import subprocess

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


class CustomBuildCommand(build_py):
    def run(self):
        subprocess.check_call(['make'])
        super().run()


setup(
    name='pyfm',
    version='0.0.1',
    author='Karl Krauth',
    author_email='karl.krauth@gmail.com',
    description='A pybind11 wrapper for libfm.',
    packages=find_packages(),
    include_package_data=True,
    cmdclass={
        'build_py': CustomBuildCommand,
    },
    install_requires=[
        'pybind11',
    ]
)
