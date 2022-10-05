#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
rocker  --nvidia --x11 \
        --name kltvo \
        --oyr-run-arg " -v /media/nigel/copel/:/data/ -v $SCRIPT_DIR/../:/root/kltvo/"  \
        kltvo:latest bash