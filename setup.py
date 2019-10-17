# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


setup(
    name='colonycounter',
    version='0.2.0',
    description='Image quantification tool for colony formation assay',
    long_description=readme,
    install_requires=['numpy',
                      'pandas',
                      'scipy',
                      'matplotlib',
                      'seaborn',
                      'scikit-image',
                      'pandas',
                      'jupyter'],
    author='Kenji Kamimoto',
    author_email='kamimoto@wustl.edu',
    url='https://',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
