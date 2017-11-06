# -*- coding: utf-8 -*-

import os
import re
import io

from setuptools import setup, find_packages

# version control performed using 
# bumpversion patch
# bumpversion dev
# bumpversion release
# bumpversion minor
# bumpversion major

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read().split('\n')

def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")
    
setup(
    name='limix_cl',
    version=find_version('limix_cl', '__init__.py'),
    description='Command line LIMIX script',
    long_description=readme,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    author='Leland Taylor',
    author_email='leland@ebi.ac.uk',
    url='https://github.com/taylordl/limix_cl',
    keywords=['limix',],
    license=license,
    install_requires=requirements,
    packages=find_packages(exclude=('bin', 'docs')),
    dependency_links=[
        'http://github.com/nfusi/qvalue.git',
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
    scripts=['bin/limix_cl'],
    include_package_data=True
)

