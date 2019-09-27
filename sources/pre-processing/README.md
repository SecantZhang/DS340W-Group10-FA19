# Script Repoitory for Data Pre-Processing

This directory contains the scripts for pre-processing the data. 

## Script and Descriptions: 

#### .bigwig Format to .txt and .wig format conversion: 
[bigwig_txt_conversion.sh](./bigwig_txt_conversion.sh)
Running this file will submit multiple jobs in parallel to do the format conversion. 
* Usage: ```sh bigwig_txt_conversion.sh```

#### Prepare Windows Files Needed for Conversion: 
[get_windows_file.sh](./get_windows_file.sh)
Downloading the chrom sizes file for hg38, convert and sort the relative file to .bed format. 
* Usage: ```sbatch get_windows_file.sh```