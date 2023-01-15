#!/bin/bash
REF_FILE=$1
RESULT_FILE=$2
evo_ape tum \
    $REF_FILE \
    $RESULT_FILE \
    --verbose \
    --plot --plot_mode=xz --plot_x_dimension index