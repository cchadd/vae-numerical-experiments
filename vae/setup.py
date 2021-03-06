#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


dist = setup(
    name="vae-pkg-CCHADD",
    version="0.0.1dev0",
    description="Exploring Variational Autoencoders",
    author="Clement Chadebec",
    author_email="clement.chadebec@gmail.com",
    license="BSD 3-Clause License",
    packages=["vae", "vae.models", "vae.trainers"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    python_requires=">=3.6",
)
