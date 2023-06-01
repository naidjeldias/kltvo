#!/bin/bash
DATA_PATH=$1
# seqs=(11 12 13 14 15 16 17 18 19 20 21)
seqs=(00 01 02 03 04 05 06 07 08 09 10)
TIME=3

echo "-------------------"
echo "Preparing to run the algorithm on KITTI sequences ..."

for SEQ in ${seqs[@]}; do
    echo "-------------------"
    echo "Runing sequence number $SEQ"
    echo "Wait for $TIME seconds ..."
    sleep $TIME
    STARTTIME=$(date +%s)
      ./stereo_kitti $DATA_PATH/$SEQ $SEQ
    ENDTIME=$(date +%s)
    echo "It takes $[$ENDTIME - $STARTTIME] seconds to complete this task..."
done