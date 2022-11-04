#!/bin/bash
DATA_PATH=$1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
rocker  --x11 \
        --devices /dev/dri \
        --name kltvo \
        --oyr-run-arg " -v $DATA_PATH:/data/ -v $SCRIPT_DIR/../:/root/kltvo/"  \
        naidjeldias/kltvo bash