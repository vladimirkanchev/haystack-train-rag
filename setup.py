import io
import os
from pathlib import Path

from setuptools import find_packages, setup

# Metadata of package
NAME = 'RAG_seven_wonders'
DESCRIPTION = 'AI RAG Q&A system about the seven ancient wonders.'
URL = 'https://github.com/vladimirkanchev/haystack-train-rag/'
AUTHOR = 'Vladimir Kanchev'
EMAIL = 'kanchev.vladimir@gmail.com'
REQUIRES_PYTHON = '>=3.10'

pwd = os.path.abspath(os.path.dirname(__file__))

# Get the list of packages to be installed 
def list_reqs(fname='requirements.txt'):
    with io.open(os.path.join(pwd, fname), encoding='utf-8') as f:
        return f.read().splitlines()

with open('VERSION') as version_file:
    _version = version_file.read().strip()

try:
    with io.open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=_version,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license='Apache License Version 2.0',
    classifiers=[
        'Programming language :: Python :: 3',
        'Programming language :: Python :: 3.10',
        'License :: OSI Aproved :: Apache License Version 2.0',
        'Operating System :: OS I'
    ],


)