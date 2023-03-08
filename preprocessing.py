import pandas as pd
import pickle
import glob
import os

class Reader():

    def __init__(self, in_path, out_path):
        self.__in_path = in_path
        self.__out_path = out_path

    def produce_pickles(self, chunksize = 10 ** 6, separator = "~"):

        with pd.read_csv(self.__in_path, separator = separator, chunksize = chunksize) as reader:
            try:
                os.makedirs(self.__out_path)
            except FileExistsError:
                # directory already exists
                pass
            for i, chunk in enumerate(reader):
                out_file = self.__out_path + "/data_{}.pkl".format(i+1)
                with open(out_file, "wb") as f:
                    pickle.dump(chunk, f, pickle.HIGHEST_PROTOCOL)
    
    def read_pickle(self, max_pickle_id = 3):
        
        data_p_files = [name for name in 
                        glob.glob(self.__out_path + "/data_*.pkl")\
                            [:max_pickle_id]]
        
        df = pd.DataFrame([])
        for i in range(len(data_p_files)):
            df = df.append(pd.read_pickle(data_p_files[i]), ignore_index=True)
        return df