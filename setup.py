#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 21:30:03 2018

@author: nhaddad
"""

from setuptools import setup, find_packages

setup(name='pydtk',
      version='0.1',
      description='A python Detector Tool Kit',
      url='https://github.com/nhaddad/pydtk',
      author='Nicolas Haddad',
      author_email='nhaddad@eso.org',
      license='MIT',
      #py_modules=["ptc", "pydtk", "utils", "utilsfunc"],
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      zip_safe=False)
