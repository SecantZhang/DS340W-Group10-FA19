#!/bin/bash
#SBATCH -N 8
#SBATCH -p RM-shared
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node 1

set -e

cd /pylon5/ci5fp2p/zzpsu/ds340w/encode_imputation/data

module load hadoop
start-hadoop.sh

hdfs dfs -mkdir spark_data
hdfs dfs -put spark_data/ml_data.csv spark_data/ml_data.csv

spark-submit spark_ml.py