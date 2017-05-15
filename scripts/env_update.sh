#!/bin/sh

PWD=`pwd`
source ${PWD}/scripts/_common.sh


source ${ENV_PATH}/bin/activate
SMART_UPDATE=${ENV_PATH}/bin/smart_update.py
echo ${ECHO_EXT} "${Red}更新依赖包: ${Gre}${SMART_UPDATE} -r requirements.txt ${RCol} >>>>>>>"

$SMART_UPDATE --update requirements.txt
