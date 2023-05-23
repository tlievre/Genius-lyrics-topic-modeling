# standard data libs
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from random import seed, sample
import pickle
import glob
import os

# data visualisation lib
import plotly.graph_objects as go
import plotly.io as pio

# change theme template for every graph below
pio.templates.default = "plotly_white"

class Loader():

    def __init__(self, in_path, out_path):
        """
        Args:
            in_path (str): csv input path.
            out_path (str): Output directory path to store the pickles.
            chunksize (int, optional): Chunksize for DataFrame reader. Defaults to 10**6. 
        """

        self.__in_path = in_path
        self.__out_path = out_path
        self.__chunksize = 10**3

    def __produce_pickles(self):
        """produce pickles by reading csv by chunksize
        """
        with pd.read_csv(self.__in_path, chunksize = self.__chunksize) as reader:
            try:
                os.makedirs(self.__out_path)
            except FileExistsError:
                # directory already exists
                pass
            for i, chunk in enumerate(reader):
                out_file = self.__out_path + "/data_{}.pkl".format(i+1)
                with open(out_file, "wb") as f:
                    pickle.dump(chunk, f, pickle.HIGHEST_PROTOCOL)

    def produce_parket(self):
        """produce parquet file reading by chunksize

        Returns:
            str: parquet_path
        """

        parquet_path = self.__out_path + "preprocess_data.parquet"

        with pd.read_csv(self.__in_path, chunksize=self.__chunksize) as csv_stream:
            try:
                os.makedirs(self.__out_path)
            except FileExistsError:
                # directory already exists
                pass
            for i, chunk in enumerate(csv_stream):
                print("Chunk", i)
                if i == 0:
                    # Guess the schema of the CSV file from the first chunk
                    parquet_schema = pa.Table.from_pandas(df=chunk).schema
                    # Open a Parquet file for writing
                    parquet_writer = pq.ParquetWriter(parquet_path,
                            parquet_schema, compression='snappy')
                # Write CSV chunk to the parquet file
                table = pa.Table.from_pandas(chunk, schema=parquet_schema)
                parquet_writer.write_table(table)
        return parquet_path
    
    def load_pickle(self, pickle_id):
        """load a pickle file by id

        Args:
            pickle_id (int): pickle id.

        Raises:
            Exception: The path of the given id isn't a file

        Returns:
            obj: DataFrame
        """
        # produce the pickles if the directory not exists or
        # if the directory is empty 
        if (not os.path.exists(self.__out_path)) or \
              (len(os.listdir(self.__out_path)) == 0):
            self.__produce_pickles()
        
        # get the file path following the pickle_id
        # given in parameter
        file_path = self.__out_path + \
            "/data_" + str(pickle_id) + ".pkl"

        if os.path.isfile(file_path):
            df = pd.read_pickle(file_path)
        else:
            raise Exception("The pickle file data_{}.pkl doesn't exist".format(pickle_id))
        return df
        

    def random_pickles(self, n_pickles = 3, init = 42, verbose = True):
        """random reader over pickles files

        Args:
            n_pickles (int, optional): number of pickles to load. Defaults to 3.
            init (int, optional): Integer given to the random seed. Defaults to 42.
            verbose (bool, optional): Print the loaded files. Defaults to True

        Raises:
            Exception: Stop the process if n_pickles exceed pickle files number.

        Returns:
            obj: pd.Dataframe
        """

        # produce the pickles if the directory not exists or
        # if the directory is empty 
        if (not os.path.exists(self.__out_path)) or \
              (len(os.listdir(self.__out_path)) == 0):
            self.__produce_pickles()

        pickle_files = [name for name in
                        glob.glob(self.__out_path + "/data_*.pkl")]
        # draw p_files        
        seed(init)

        if n_pickles <= 6:
            random_p_files = sample(pickle_files, n_pickles)
        else:
            raise Exception("The parameter n_pickles (" +
                            "{}) exceed the numbers of pickle files ({})"\
                                .format(n_pickles, len(pickle_files)))
        # print the drawed files
        if verbose:
            print("Loaded pickles:")
            for p in random_p_files:
                print(p)

        # load random pickles file
        df_list = [pd.read_pickle(p) for p in random_p_files]

        # create the dataframe by concatenate the previous
        # dataframes list
        df = pd.concat(df_list, ignore_index = True)
        return df
    
    # get some information about the pickle data
def pickle_informations(self):
    rows = []
    for i in range(1, len(os.listdir('data')) + 1):
        df = self.load_pickle(i)
        rows.append(len(df))
        del df
    return rows

def add_bar(self, i, y1, y2, color):
    """add bar to barchart representation

    Args:
        i (int): integer label of the added bar
        y1 (int): lower bound date
        y2 (int): upper bound date
        color (str): color of the added bar

    Returns:
        obj, obj: plotly objects.
    """
    df = self.load_pickle(i)
    df = df[(df.year >= y1) & (df.year <= y2)]
    df_year = df.groupby(['year']).size().reset_index(name='count')
    new_bar = go.Bar(
                x = df_year.year.values,
                y = df_year['count'].values,
                name = 'data_'+ str(i),
                marker = {'color' : color})
    new_trend = go.Scatter(
                x = df_year.year.values,
                y = df_year['count'].values,
                mode="lines",
                line={'color' : color,
                    'width' : 0.5},
                showlegend=False)
    del df_year, df
    return new_bar, new_trend


def multi_barplot(self, year1, year2, colors):
    """get a multi barplot vizualisation given loaded data

    Raises:
        Exception: Raise exception if the colors list
        doesn't fit with the number batches

    Returns:
        obj: plotly object
    """
    # create a empty plotly.Figure object
    fig = go.Figure() 
    # compute the batch number
    n_batch = len(os.listdir('data'))
    # test the color list feed in argument
    # fit well with the batch number
    if n_batch > len(colors):
        raise Exception(
            "The colors list size({})doesn't ".format(len(colors)) +
            "fit with the number of data".format(n_batch))
    for i in range(1, n_batch + 1):
        fig.add_traces((self.add_bar(i, year1, year2, colors[i-1])))
    fig.update_layout(
        title = "Data distribution over years ({} - {})"
            .format(year1, year2),
        xaxis_title="years",
        yaxis_title="title",
        legend_title="Data batch")
    return fig