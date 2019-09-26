# DS340W-Group10-FA19
Private Repository for DS340W Project, owned by [Zheng Zhang](http://zheng-zhang.com) and Madison Novak

## Essential Resources Addresses: 
### I. Code: 
* [Data-Preprocessing](./sources/pre-processing)

### II. Data: 
* Original data in .bigwig format: 
    * Training: ```/pylon5/ci5fp2p/zzpsu/ds340w/encode_imputation/data/training_data```
    * Validation: ```/pylon5/ci5fp2p/zzpsu/ds340w/encode_imputation/data/validation_data```
* Converted data: 
    * txt file: In the ```txt_dir``` under the ```training_data``` and ```validation_data``` directory. 
    * wig file: In the ```wig_dir``` under the ```training_data``` and ```validation_data``` directory. 

## Commonly Shared Software Location and Usage: 
1. bigWigToWig: Converts the ```.bigwig``` format to ```.wig``` format. 
    + Address: ```/pylon5/ci5fp2p/zzpsu/ds340w/encode_imputation/software/bigWigToWig```
    + Usage: ```bigWigToWig in.bigWig out.wig```
    + Options: 
        + -chrom=chr1 - if set restrict output to given chromosome
        + -start=N - if set, restrict output to only that over start
        + -end=N - if set, restict output to only that under end
        + -udcDir=/dir/to/cache - place to put cache for remote bigBed/bigWigs