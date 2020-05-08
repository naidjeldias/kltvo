#!/bin/bash
SEQ=03
NUM=1
TIME=3
echo "-------------------"
echo "Preparing to run the algorithm in sequence $SEQ, $NUM times ..."

COUNTER=0
echo "Creating results folder ..."
cd results/
mkdir $SEQ
cd ..

echo "Creating stats folder ..."
cd stats
mkdir $SEQ
cd ..

while [ $COUNTER -lt $NUM ]; do
    echo "-------------------"
    echo "Run number $COUNTER"
    echo "Wait for $TIME seconds ..."
    RESULT_PATH="results/$SEQ/"
    STATS_PATH="stats/$SEQ/"
    FILE_NAME="$COUNTER"
    sleep $TIME
    STARTTIME=$(date +%s)
      ./stereo_kitti $SEQ $RESULT_PATH $FILE_NAME $STATS_PATH
    ENDTIME=$(date +%s)
    echo "It takes $[$ENDTIME - $STARTTIME] seconds to complete this task..."
    echo "Evaluating on EVO tool ..."
    cd ../../EVO_VO/evo/
    RESULT_FULL_PATH="../../CLionProjects/klt-vo/$RESULT_PATH$FILE_NAME.txt"
    DATASET_PATH="../../KITTI_DATASET/poses/poses/$SEQ.txt"
    evo_ape kitti  $DATASET_PATH $RESULT_FULL_PATH -v 
    cd ../../CLionProjects/klt-vo/
    let COUNTER=COUNTER+1
done
