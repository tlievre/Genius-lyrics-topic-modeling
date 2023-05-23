from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import min, col
from modules.loader import Loader # relative import relate to notebook
import math

@udf(IntegerType())
def decade(year):
    return int(math.trunc(year / 10) * 10)
# decadeUDf = udf(lambda z: decade(z), IntegerType())

class SparkSPreprocessor():

    def __init__(self, input_path, parquet_path, driver_memory = "20g"):
        
        # produce parket file
        parquet_file = Loader(in_path=input_path, out_path=parquet_path).produce_parket()

        spark = SparkSession.builder \
            .appName("MyApp") \
            .master("local[2]") \
            .config("spark.driver.memory", driver_memory) \
            .getOrCreate()
        
        self.__df = spark.read.parquet(parquet_file)


    def preprocess_data(self, begin = 1960, end = 2023, freq = 1,
                    seed = 117, write_csv = True,
                    csv_path = "./data/preprocessed_data.csv"):
        
        # add decade column
        self.__df.withColumn("decade", decade(self.__df.year))

        # get our data of interest by
        # filtering:
        #   - bounding years release (1960 - 2023)
        #   - English song
        #   - oultlier artist 'Genius Translations'
        filter_query = self.__df.filter(
                self.__df.year.between(begin, end) & 
                (self.__df.language == 'en') &
                (self.__df.artist != 'Genius English Translations')) \
            .withColumn("decade", decade(self.__df.year)) \
            .na.drop(subset=["title"])
        
        # compute decade frequency
        decade_freq = filter_query.groupBy("decade").count()

        # compute the smallest decade frequency
        n_sample = decade_freq.select(min("count")) \
            .collect()[0]["min(count)"]

        # compute list fraction
        frac = decade_freq.withColumn("required_n", freq * n_sample / col("count")) \
            .drop("count").rdd.collectAsMap()
        
        result_query = filter_query.sampleBy("decade", frac, seed)

        # create pandas result       
        pandas_df = result_query.toPandas()

        # writing a csv file
        if write_csv:
            pandas_df.to_csv(csv_path, index = False)
        return pandas_df