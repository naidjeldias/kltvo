#!/bin/bash
SEQ=$1

FILE="/root/kltvo/examples/kitti/results/KITTI_${SEQ}_KLTVO.txt"
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    echo "$FILE does not exist."
    echo "Running program with defined sequence"
    /root/kltvo/./stereo_kitti /data/KITTI-dataset/data_odometry_gray/dataset/sequences/$SEQ/ $SEQ
fi

echo "Running benchmark with evo package"
FILE="/root/kltvo/examples/kitti/results/KITTI_${SEQ}_KLTVO.tum"
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
echo "Converting KLTVO result to TUM format"
python3 /root/kltvo/scripts/kitti_to_tum.py /root/kltvo/examples/kitti/results/KITTI_${SEQ}_KLTVO.txt \
        /data/KITTI-dataset/data_odometry_gray/dataset/sequences/$SEQ/times.txt \
        /root/kltvo/examples/kitti/results/KITTI_${SEQ}_KLTVO.tum
fi

FILE="/data/KITTI-dataset/data_odometry_poses/tum_format/${SEQ}.tum"
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
echo "Converting KITTI GT to TUM format"
python3 /root/kltvo/scripts/kitti_to_tum.py /data/KITTI-dataset/data_odometry_poses/dataset/poses/${SEQ}.txt \
        /data/KITTI-dataset/data_odometry_gray/dataset/sequences/$SEQ/times.txt \
        /data/KITTI-dataset/data_odometry_poses/tum_format/${SEQ}.tum
fi

echo "Translation part"
DIR="/root/kltvo/examples/kitti/results/${SEQ}_trans"
if [ -d "$DIR" ];
then
    echo "$DIR directory exists."
else
mkdir $DIR
evo_rpe tum \
    /data/KITTI-dataset/data_odometry_poses/tum_format/${SEQ}.tum \
    /root/kltvo/examples/kitti/results/KITTI_${SEQ}_KLTVO.tum \
    --pose_relation trans_part \
    --delta 1 \
    --delta_unit f \
    --plot_x_dimension index \
    --save_results $DIR/${SEQ}_trans.zip \
    --save_plot $DIR/

unzip $DIR/${SEQ}_trans.zip -d $DIR
fi

echo "Rotation part"
DIR="/root/kltvo/examples/kitti/results/${SEQ}_rot"
if [ -d "$DIR" ];
then
    echo "$DIR directory exists."
else
mkdir $DIR
evo_rpe tum \
    /data/KITTI-dataset/data_odometry_poses/tum_format/${SEQ}.tum \
    /root/kltvo/examples/kitti/results/KITTI_${SEQ}_KLTVO.tum \
    --pose_relation rot_part \
    --delta 1 \
    --delta_unit f \
    --plot_x_dimension index \
    --save_results $DIR/${SEQ}_rot.zip \
    --save_plot $DIR/

unzip $DIR/${SEQ}_rot.zip -d $DIR
fi

echo "Stats data augmentation"
echo "Adding translation RPE"

python3 /root/kltvo/scripts/plot_data.py \
        -f /root/kltvo/examples/kitti/stats/KITTI_${SEQ}_STATS.csv \
        --error_file /root/kltvo/examples/kitti/results/${SEQ}_trans/error_array.npy \
        --fuse_data --column_name rpe_trans \
        -o /root/kltvo/examples/kitti/stats/KITTI_${SEQ}_STATS_extended.csv

echo "Adding rotation RPE"

python3 /root/kltvo/scripts/plot_data.py \
        -f /root/kltvo/examples/kitti/stats/KITTI_${SEQ}_STATS_extended.csv \
        --error_file /root/kltvo/examples/kitti/results/${SEQ}_rot/error_array.npy \
        --fuse_data --column_name rpe_rot \
        -o /root/kltvo/examples/kitti/stats/KITTI_${SEQ}_STATS_extended.csv

echo "Computing correlation matrix"
python3 /root/kltvo/scripts/plot_data.py \
        -f /root/kltvo/examples/kitti/stats/KITTI_${SEQ}_STATS_extended.csv \
        --compute_df_correlation --output /root/kltvo/examples/kitti/stats/KITTI_${SEQ}_CORR_MAT.png
