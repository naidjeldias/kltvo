#!/bin/bash
SEQ=00
NUM=30
TIME=3
echo "-------------------"
echo "Preparing to run the algorithm in sequence $SEQ, $NUM times ..."

COUNTER=0
echo "Creating results folder ..."
cd results/
mkdir $SEQ
cd ..

while [ $COUNTER -lt $NUM ]; do
    echo "-------------------"
    echo "Run number $COUNTER"
    echo "Wait for $TIME seconds ..."
    RESULT_PATH="results/$SEQ/"
    FILE_NAME="$COUNTER.txt"
    sleep $TIME
    STARTTIME=$(date +%s)
      ./stereo_kitti $SEQ $RESULT_PATH $FILE_NAME
    ENDTIME=$(date +%s)
    echo "It takes $[$ENDTIME - $STARTTIME] seconds to complete this task..."
    echo "Evaluating on EVO tool ..."
    cd ../../EVO_VO/evo/
    RESULT_FULL_PATH="../../CLionProjects/klt-vo/$RESULT_PATH$FILE_NAME"
    DATASET_PATH="../../KITTI_DATASET/poses/poses/$SEQ.txt"
    evo_rpe kitti  $DATASET_PATH $RESULT_FULL_PATH -v --delta 1
    cd ../../CLionProjects/klt-vo/
    let COUNTER=COUNTER+1
done
