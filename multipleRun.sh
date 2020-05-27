#!/bin/bash
SEQ=00
NUM=30
TIME=5
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
    cd ../../EVO_VO/evo/
    echo "Converting to TUM format ..."
    TUM_PATH="../../CLionProjects/klt-vo/results/tum/$FILE_NAME.tum"
    KITTI_TIME_PATH="../../KITTI_DATASET/dataset/sequences/$SEQ/times.txt"
    KITTI_TUM_PATH="../../KITTI_DATASET/poses/tum/$SEQ.tum"
    RESULT_FULL_PATH="../../CLionProjects/klt-vo/$RESULT_PATH$FILE_NAME.txt"
    python contrib/kitti_poses_and_timestamps_to_trajectory.py $RESULT_FULL_PATH $KITTI_TIME_PATH $TUM_PATH
    echo "TUM format saved on $TUM_PATH"
    echo "Evaluating on EVO tool ..."
    DATASET_PATH="../../KITTI_DATASET/poses/poses/$SEQ.txt"
    echo "Computing ATE ..."
    evo_ape kitti  $DATASET_PATH $RESULT_FULL_PATH -v
    echo "Computing RPE translation part ..."
    evo_rpe tum $KITTI_TUM_PATH $TUM_PATH -v
    echo "Computing RPE rotation part ..."
    evo_rpe tum $KITTI_TUM_PATH $TUM_PATH -v --pose_relation angle_deg
    cd ../../CLionProjects/klt-vo/
    let COUNTER=COUNTER+1
done
