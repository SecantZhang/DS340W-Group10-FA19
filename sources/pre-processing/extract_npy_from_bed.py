import numpy as np
import glob
import os
import datetime
import re
from multiprocessing import Pool


log_file = None


def encapsulate_data(true_path):
    """
    Function for encapsulate the numeric data to the hdf5 data type.
    :param path: Relative path to the folder.
    :param hf_data: hdf5 adt
    :param log_file: Text object for writing the execution logs.
    :return: nothing
    """
    prefix = os.path.basename(true_path).split("_")[0]
    cell_type = str(prefix.split("M")[0].split("C")[1])
    mark_type = str(prefix.split("M")[1])
    print(true_path, "| Cell: ", cell_type, " | Mark: ", mark_type)
    log_file.write("\n----------------------------------------------\n")
    log_file.write("Start processing the bed file with PREFIX: " + prefix + " At time: " + str(datetime.datetime.now()) + '\n')
    log_file.flush()

    # Open the file.
    with open(true_path) as true_file:
        log_file.write("PREFIX Name: " + prefix + " | Start processing bed files. \n")
        chrom_dict = {}
        # Iterate through the file and write each line to a list and convert it to the numpy array.
        log_file.write("Iterating through the bed file \n")
        log_file.flush()
        for index, line in enumerate(true_file):
            split_line = line.split("\t")
            chrom = split_line[0]  # chrom name

            if re.match("^chr([0-2][0-9]|[0-9]|X|x|Y|y)$", chrom) is None:
                continue

            # Check if the chrom is already existed in the dictionary. If not existed, create a key and array value.
            if chrom not in chrom_dict:
                chrom_dict[chrom] = []
            else:
                # Check if the split_line contains 4 values.
                if split_line.__len__() != 4:
                    log_file.write("ERROR - Corrupted array with index: " + str(index) + "\n")
                    log_file.write("----Original array: " + str(line) + "\n")
                    # Append error message -3
                    chrom_dict[chrom].append([-3, -3])
                    continue
                else:
                    if split_line[3] == '.\n':
                        # Append error message 0
                        split_line[1:4] = int(split_line[1]), int(split_line[2]), 0
                        chrom_dict[chrom].append([split_line[1], split_line[3]])
                    elif split_line[3] == '\n':
                        # Append error message -1
                        split_line[1:4] = int(split_line[1]), int(split_line[2]), -1
                        chrom_dict[chrom].append([split_line[1], split_line[3]])
                    elif isinstance(float(split_line[3]), float):
                        split_line[1:4] = int(split_line[1]), int(split_line[2]), float(split_line[3])
                        chrom_dict[chrom].append([split_line[1], split_line[3]])
                    else:
                        # Append error message -2
                        split_line[1:4] = int(split_line[1]), int(split_line[2]), -2
                        chrom_dict[chrom].append([split_line[1], split_line[3]])

        try:
            for chrom in chrom_dict:
                if os.path.isfile("bed_npy_dir/{}_25-{}.npy".format(prefix, chrom)):
                    log_file.write("Ignoring the file {}_25-{}.npy. ".format(prefix, chrom))
                    continue
                else:
                    cur_data = np.array(chrom_dict[chrom])
                    # Save the numpy array into the bed_npy_directory as the npy format.
                    np.save("bed_npy_dir/{}_25-{}.npy".format(prefix, chrom), cur_data)
        except Exception as e:
            log_file.write("Errors when converting np_ary to numpy array.\n")
            log_file.write(e)

    log_file.write("End processing the bed file with PREFIX: " + prefix + " At time: "
                    + str(datetime.datetime.now()) + '\n')
    log_file.write("\n----------------------------------------------\n")
    log_file.flush()


if __name__ == "__main__":
    # # Initialize the hdf5 file object.
    # hf_train = h5py.File("output/hf_training_bed.hdf5", "w")
    # encapsulate_data("bed_file/*25.bed", hf_train)
    # hf_train.close()
    # log_file = open("validation_log.txt", "w+")
    # hf_valid = h5py.File("hdf5_bed_output/hf_validate_bed.hdf5", "w")
    # encapsulate_data("validation_data_new/C09M20_25.bed", hf_valid, log_file)
    # log_file.close()
    # hf_valid.close()

    global log_file = open("log_dir/training_log2.txt", "w+")
    path_list = list(glob.glob("*_25.bed"))
    pool = Pool()
    pool.map(encapsulate_data, path_list)
    pool.close()
    log_file.close()