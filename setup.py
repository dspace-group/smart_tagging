# Copyright 2022, dSPACE GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you must not use this software except in compliance with the License. This
# software is not fully developed or tested. It is distributed free of charge
# and without any consideration. The software is provided "as is" in the hope
# that it may be useful to other users, but without any warranty of any kind,
# either express or implied. See the License for the specific language
# governing permissions and limitations under the License.


from setuptools import setup, find_packages

# load the README file and use it as the long_description for PyPI
with open('README.md', 'r') as f:
    readme = f.read()

# set the version info, imports __version__
__version__ = ''
exec(open('smart_tagging/version.py').read())

setup(
    name='smart_tagging',
    description=(
        "Smart Tagging is a collection of RTMaps diagrams which "
        "deploy neural network based algorithms, e.g. for object detection."
    ),
    long_description=readme,
    long_description_content_type='text/markdown',
    version=__version__,
    author='dSPACE AI Team',
    url='www.dspace.com',
    license='Apache v2.0',
    license_files='LICENSE.txt',
    classifiers=[
        'Development Status :: 4 - Beta',

        'License :: OSI Approved :: Apache Software License',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',

        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/dspace-group/smart_tagging/issues',
        'Source': 'https://github.com/dspace-group/smart_tagging',
    },
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6, <4',
    install_requires=[
        'requests>=2.27.0',
        'tensorflow~=2.4.0',
    ],
)
