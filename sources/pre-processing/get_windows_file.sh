#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 5:00:00
#SBATCH --ntasks-per-node 1

set -e

module load bedtools/2.25.0
module load gcc/9.1.0

CURRENTDATETIME=`date +"%Y-%m-%d %T"`
echo $CURRENTDATETIME

ENCODE_PATH=/pylon5/ci5fp2p/zzpsu/ds340w/encode_imputation/data
cd $ENCODE_PATH/bin_misc

# genome windows file
if [ ! -e hg38.chrom.sizes ]
	then
		curl http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes -o hg38.chrom.sizes
fi
if [ ! -e hg38_25_windows.bed ]
	then
		bedtools makewindows -g hg38.chrom.sizes -w 25 > hg38_25_windows.bed
fi
if [ ! -e hg38_25_windows_sorted.bed ]
	then
		sort -k 1,1 -k2,2n hg38_25_windows.bed > hg38_25_windows_sorted.bed
fi