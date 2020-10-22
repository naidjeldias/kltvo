#!/bin/bash
seqs=(11 12 13 14 15 16 17 18 19 20 21)
TIME=5

echo "-------------------"
echo "Preparing to run the algorithm on KITTI benchmark sequences ..."

echo "Creating results folder ..."
cd results/
mkdir KITTI
cd ..

echo "Creating stats folder ..."
cd stats
mkdir KITTI
cd ..


for SEQ in ${seqs[@]}; do
    echo "-------------------"
    echo "Runing sequence number $SEQ"
    echo "Wait for $TIME seconds ..."
    RESULT_PATH="results/KITTI/"
    STATS_PATH="stats/KITTI/"
    FILE_NAME="$SEQ"
    sleep $TIME
    STARTTIME=$(date +%s)
      ./stereo_kitti $SEQ $RESULT_PATH $FILE_NAME $STATS_PATH
    ENDTIME=$(date +%s)
    echo "It takes $[$ENDTIME - $STARTTIME] seconds to complete this task..."
done