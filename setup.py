import glob
import subprocess

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


class CustomBuildCommand(build_py):
    def run(self):
        subprocess.check_call(['make'])
        super().run()


setup(
    name='wpyfm',
    version='0.1.5',
    author='Karl Krauth',
    author_email='karl.krauth@gmail.com',
    description='A pybind11 wrapper for libfm.',
    download_url= 'https://github.com/Karl-Krauth/libfm/archive/v0.1.5.tar.gz',
    packages=find_packages(),
    include_package_data=True,
    cmdclass={
        'build_py': CustomBuildCommand,
    },
    install_requires=[
        'pybind11',
    ]
)
