import sys
import numpy as np
import glob
import os


def down_size(lt, output_path, avg, name): 
    prefix, chrom = extract_prefix(lt, avg)
    if chrom == "21": 
        print("Aggregating {}_{}_{}.npy".format(prefix, name, chrom))
        splits = 40 # Merge 25 windows size to 1000 windows size. 
        with open("/pylon5/ci5fp2p/zzpsu/ds340w/encode_imputation/data/training_data/txt_dir/avg/" + file_path) as file: 
            curr_file = [float(line.strip()) for line in file]
            # output_li = [sum(curr_file[i:i+splits])/splits if i+splits < len(curr_file) else sum(curr_file[i:]/len(curr_file)-i for i in range(len(curr_file) - splits + 1)]
            leng = len(curr_file)
            output_li = []
            for i in range(0,leng,splits): 
                if i + splits < leng: 
                    output_li.append(mean(curr_file[i:i+splits]))
                else: 
                    output_li.append(mean(curr_file[i:]))
            print("File Statistics: length - {}, sample data point (1500-1510) - [{}]".format(len(output_li), output_li[1500:1510]))
            np.save(output_path + "/{}_{}_{}.npy".format(prefix, name, chrom), output_li)
            print("File Saved: {}_{}_{}.npy".format(prefix, name, chrom))

def mean(li): 
    return sum(li)/len(li)

def clean(lt, output_path, avg, name): 
    prefix, chrom = extract_prefix(lt, avg)
    output_li = [0 if x == '.' else x for x in lt]
    np.save(output_path + "/{}_{}_{}.npy".format(prefix, name, chrom), output_li)

def extract_prefix(path, avg=False): 
    if avg: 
        prefix, chrom = path.split("_")[1:3]
    else: 
        prefix = path.split("_")[0]
        chrom = "21"
    return [prefix, chrom]

if __name__ == "__main__": 
    # file_path = sys.argv[1]
    # output_path = sys.argv[2]
    # option = sys.argv[3]
    # name = sys.argv[4]
    # avg = sys.argv[5]
    # lt = [line.rstrip('\n') for line in open(file_path)]
    output_path = "/pylon5/ci5fp2p/zzpsu/ds340w/encode_imputation/data/training_data/txt_dir/window_size_1000"
    option = "down"

    if option == "down": 
        for path in glob.glob("/pylon5/ci5fp2p/zzpsu/ds340w/encode_imputation/data/training_data/txt_dir/avg/*.txt"): 
            file_path = os.path.basename(path)
            down_size(file_path, output_path, True, "training")
    # elif option == "clean": 
    #     clean(file_path, output_path)
    print("Done")