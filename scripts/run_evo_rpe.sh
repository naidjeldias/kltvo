#!/bin/bash
REF_FILE=$1
RESULT_FILE=$2
POSE_RELATION=$3 # trans_part, rot_part
evo_rpe tum \
    $REF_FILE \
    $RESULT_FILE \
    --pose_relation $POSE_RELATION \
    --delta 1 \
    --delta_unit f \
    --plot --plot_mode=xz --plot_x_dimension index