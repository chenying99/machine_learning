#!/bin/sh

PWD=`pwd`
source ${PWD}/scripts/_common.sh

echo ${ECHO_EXT} "${Red}初始化开始: ${Gre}${ENV_PATH}${RCol} >>>>>>>"

#
# 线上服务器的Python都为2.7, 且位置在: /usr/local/python2.7
#
if [ -f /usr/local/python27/bin/virtualenv ]; then
    VIRTUAL_ENV=/usr/local/python27/bin/virtualenv
else
    VIRTUAL_ENV=`which virtualenv`
fi


if [ ! -d ${ENV_PATH} ]; then
    ${VIRTUAL_ENV} ${ENV_PATH}
fi

source ${ENV_PATH}/bin/activate

SITE_CUSTOMIZE="${ENV_PATH}/lib/python2.7/site-packages/sitecustomize.py"
if [ ! -f ${SITE_CUSTOMIZE}  ]; then
    cat >> ${SITE_CUSTOMIZE} << "EOF"
# -*- coding:utf-8 -*-
#
# 设置系统的默认编码, 这样utf-8和unicode之间就可以自由转换了；否则系统默认的编码为ascii
#
import sys
sys.setdefaultencoding('utf-8')
EOF
fi


pip install pip

# Ubuntu/Linux 64-bit, CPU only, Python 2.7

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0-cp27-none-linux_x86_64.whl
# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Installing from sources" below.
# export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp27-none-linux_x86_64.whl
# Mac OS X, CPU only, Python 2.7:
# export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.0-py2-none-any.whl
pip install --upgrade $TF_BINARY_URL

deactivate

echo ${ECHO_EXT} "${Red}初始化完毕: ${Gre}${ENV_PATH}${RCol} <<<<<<"
