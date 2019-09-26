set -e 

WORKING_DIR=/pylon5/ci5fp2p/zzpsu/ds340w/encode_imputation/data
TRAINING_DIR=$WORKING_DIR/training_data/ 
VALIDATION_DIR=$WORKING_DIR/validation_data/
S=/pylon5/ci5fp2p/zzpsu/ds340w/encode_imputation/software

mkdir -p ${TRAINING_DIR}wig_dir/wig
mkdir -p ${TRAINING_DIR}wig_dir/sorted_wig
mkdir -p ${TRAINING_DIR}bed_dir
mkdir -p ${TRAINING_DIR}txt_dir
mkdir -p ${TRAINING_DIR}shell_script

mkdir -p ${VALIDATION_DIR}wig_dir/wig
mkdir -p ${VALIDATION_DIR}wig_dir/sorted_wig
mkdir -p ${VALIDATION_DIR}bed_dir
mkdir -p ${VALIDATION_DIR}txt_dir
mkdir -p ${VALIDATION_DIR}shell_script


cd $TRAINING_DIR

for f in *.bigwig
do 
    PREFIX=${f%.bigwig}
    if [ ! -e ${TRAINING_DIR}txt_dir/${PREFIX}_25.txt ]
        then
            CURRENTDATETIME=`date +"%Y-%m-%d %T"`
            echo "------------------------- $f Processing File."
            echo "#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 5:00:00
#SBATCH --ntasks-per-node 1

module load bedtools/2.25.0
module load gcc/9.1.0

echo 'Job Started at ${CURRENTDATETIME}.'

cd ${TRAINING_DIR}

echo ${PREFIX}

if [ ! -e wig_dir/wig/$PREFIX.wig ]
	then
		$S/bigWigToWig $f wig_dir/wig/$PREFIX.wig
fi

if [ ! -e wig_dir/sorted_wig/${PREFIX}_sorted.wig ]
	then
		sort -k 1,1 -k2,2n wig_dir/wig/$PREFIX.wig > wig_dir/sorted_wig/${PREFIX}_sorted.wig
fi

if [ ! -e bed_dir/${PREFIX}_25.bed ]
	then 
		bedtools map -a ../hg38_25_windows_sorted.bed -b wig_dir/sorted_wig/${PREFIX}_sorted.wig -c 4 -o mean > bed_dir/${PREFIX}_25.bed
fi 

if [ ! -e txt_dir/${PREFIX}_25.txt ]
    then
        bedtools map -a ../hg38_25_windows_sorted.bed -b wig_dir/sorted_wig/${PREFIX}_sorted.wig -c 4 -o mean | awk '\$1 == \"chr21\" {print \$4}' > txt_dir/${PREFIX}_25.txt
fi

echo \"Done Processing the script\"
            " > ${TRAINING_DIR}shell_script/${PREFIX}_txt_train.sh 
            sbatch ${TRAINING_DIR}shell_script/${PREFIX}_txt_train.sh 

            echo "------------------------- ${PREFIX}_txt_train.sh ------- Done submitting the training job."
    else
        echo "------------------------- Skipped training file ---- ${PREFIX}"
    fi

done

cd $VALIDATION_DIR
echo "------------------------- Change Directory to validation."

for f in *.bigwig
do 
    PREFIX=${f%.bigwig}
    if [ ! -e ${VALIDATION_DIR}txt_dir/${PREFIX}_25.txt ]
        then
            CURRENTDATETIME=`date +"%Y-%m-%d %T"`
            echo "------------------------- $f Processing File."
            echo "#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 5:00:00
#SBATCH --ntasks-per-node 1

module load bedtools/2.25.0
module load gcc/9.1.0

echo 'Job Started at ${CURRENTDATETIME}.'

cd ${VALIDATION_DIR}

echo ${PREFIX}

if [ ! -e wig_dir/wig/$PREFIX.wig ]
	then
		$S/bigWigToWig $f wig_dir/wig/$PREFIX.wig
fi

if [ ! -e wig_dir/sorted_wig/${PREFIX}_sorted.wig ]
	then
		sort -k 1,1 -k2,2n wig_dir/wig/$PREFIX.wig > wig_dir/sorted_wig/${PREFIX}_sorted.wig
fi

if [ ! -e bed_dir/${PREFIX}_25.bed ]
 	then 
 		bedtools map -a ../hg38_25_windows_sorted.bed -b wig_dir/sorted_wig/${PREFIX}_sorted.wig -c 4 -o mean > bed_dir/${PREFIX}_25.bed
fi 

if [ ! -e txt_dir/${PREFIX}_25.txt ]
    then
        bedtools map -a ../hg38_25_windows_sorted.bed -b wig_dir/sorted_wig/${PREFIX}_sorted.wig -c 4 -o mean | awk '\$1 == \"chr21\" {print \$4}' > txt_dir/${PREFIX}_25.txt
fi
echo \"Done Processing the script\"
            " > shell_script/${PREFIX}_txt_valid.sh 
            sbatch shell_script/${PREFIX}_txt_valid.sh 

            echo "------------------------- ${PREFIX}_txt_valid.sh ------- Done submitting the validation job."
    else
        echo "------------------------- Skipped validation file ---- ${PREFIX}"
    fi

done
