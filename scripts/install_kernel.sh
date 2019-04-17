#!/bin/bash

#print what you are doing:
set -x

#create kernel dir
mkdir -p ${HOME}/.local/share/jupyter/kernels

#cp kernel to jupyter kernels
cp /usr/common/software/tensorflow/intel-tensorflow/1.13.0-py36-dev/share/jupyter/kernels/python3/kernel.json ${HOME}/.local/share/jupyter/kernels/ 
