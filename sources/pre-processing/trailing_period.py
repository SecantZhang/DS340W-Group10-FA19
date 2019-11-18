import numpy as np 
import glob
import os


if __name__ == "__main__": 
    general_path_li = ["/Users/Michavillson/Documents/PROJECTS/DS340W-Group10-FA19/sources/ML_model/sample_data/valdiation/",
                    "/Users/Michavillson/Documents/PROJECTS/DS340W-Group10-FA19/sources/ML_model/sample_data/avocado/"]
    for general_path in general_path_li:
        for path in glob.glob(general_path + "*.txt"): 
            prefix = os.path.basename(path).split(".")[0]
            with open(path) as file: 
                output_li = [float(line.strip()) if line.strip() != "." else 0.0 for line in file]
                np.save("{}npy/{}.npy".format(general_path, prefix), output_li)