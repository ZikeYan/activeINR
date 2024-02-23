#!/usr/bin/env python

from setuptools import setup
import shlex
import subprocess


def git_version():
    cmd = "git describe --exact-match --tags $(git log -n1 --pretty='%h')"
    try:
        version = subprocess.check_output(cmd, shell=True).decode()
    except:
        version = "0.0"
    return version

version = git_version()

setup(
    name='activeINR',
    version=version,
    author='Zike Yan',
    author_email='yanzike@air.tsinghua.edu.cn',
    py_modules=[]
)
