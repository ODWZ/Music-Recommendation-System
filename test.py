#Part 2: Testing
#Usage:
#    $ spark-submit --conf spark.driver.memory=4g --conf spark.executor.memory=16g baseline_no_test.py hdfs:/user/bm106/pub/project/cf_train.parquet hdfs:/user/bm106/pub/project/cf_validation.parquet hdfs:/user/bm106/pub/project/cf_test.parquet True


# import libraries 
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark import SparkConf
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.sql.functions import concat, col, lit
import numpy as np
import itertools
from itertools import *
from pyspark import SparkContext
sc =SparkContext()


## train_data = "hdfs:/user/bm106/pub/project/cf_train.parquet"
## val_data = "hdfs:/user/bm106/pub/project/cf_validation.parquet"
## test_data = "hdfs:/user/bm106/pub/project/cf_test.parquet"

def main(spark, train_data, test_data, rank, regParam, alpha, lower_bound = None, log_compression = None):
    '''Main routine for supervised training
    Parameters
    ----------
    spark : SparkSession object
    train_data : string, path to the training parquet file to load
    val_data: string, path to the validation parquet file to load
    test_data: string, path to the testing parquet file to load
    downsample: TRUE or FALSE. To indicate if we should downsample the data or not
    '''
    
    ### read in the files 
    train = spark.read.parquet(train_data)
    test = spark.read.parquet(test_data)
    
    ### if down-sample: down-sample train data to random 0.1%
    if log_compression != None: # log-compression
        train = train.withColumn("log_count", log("count")) # apply log-compression

    if lower_bound != None:
        train = train.filter(train["count"]>int(lower_bound)) # filter out count data rows lower than the lower bound

    ### transform dataframe columns: user_id, track_id from string to float and put them in the pipeline
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_indexed", handleInvalid='skip')
    track_indexer = StringIndexer(inputCol="track_id", outputCol="track_id_indexed", handleInvalid='skip')
    pipeline = Pipeline(stages=[user_indexer, track_indexer])
    indexing_model = pipeline.fit(train) #learn (return: pipeline model)

    ### transform the datasets and create the view 
    train = indexing_model.transform(train) # return a dataframe with new columns 
    train.createOrReplaceTempView("train")
    test.createOrReplaceTempView("test")

    # group by user_id, aggregate track_id_indexed for test data
    test_groupby = spark.sql("select user_id_indexed, collect_list(track_id_indexed) track_id_indexed_collections from test group by user_id_indexed")
    test_groupby.createOrReplaceTempView("test_groupby")

    rank_ = rank
    regParam_ = regParam
    alpha_ = alpha

    if log_compression != None:
        ratingCol = "log_count"
    else:
        ratingCol = "count"
    
    als = ALS(rank=float(rank_), regParam=float(regParam_), alpha=float(alpha_), implicitPrefs=True, userCol="user_id_indexed", itemCol="track_id_indexed", ratingCol=ratingCol, coldStartStrategy="drop")
    model = als.fit(train) # fit the pipeline onto training data
    # get top 500 recommendations
    userRecs = model.recommendForAllUsers(500) # return: dataframe (columns: user_id_indexed, recommendations) 
                                                    # [('user_id_indexed', 'int'), ('recommendations', 'array<struct<track_id_indexed:int,rating:float>>')]
    userRecs = userRecs.select(userRecs.user_id_indexed, userRecs.recommendations.track_id_indexed.alias("pred_list")) # with track_id_indexed only, no track_id
    userRecs.createOrReplaceTempView("userRecs") # create temporary view
        
    combined_df = spark.sql('''select test_groupby.user_id_indexed user_id_indexed, userRecs.pred_list pred_list, 
        test_groupby.track_id_indexed_collections track_id_indexed_collections from userRecs inner join test_groupby on test_groupby.user_id_indexed = userRecs.user_id_indexed''') # combine dfs wrg to user_id_indexed

    # use ranking metrics for evaluations
    predLabelsTuple = combined_df.rdd.map(lambda r: (r.pred_list, r.track_id_indexed_collections)) # result: tuple
    # predictionAndLabels = sc.parallelize([predLabelsTuple.collect()])
    metrics = RankingMetrics(predLabelsTuple)
    MAP = metrics.meanAveragePrecision
    precision_at_500 = metrics.precisionAt(500)
    # print out validation evaluation result
    print("---------------------------------------")
    print("configs: \n")
    print("rank = " + str(rank_) + " , regParam = " + str(regParam_) + " , alpha = " + str(alpha_))
    print("\n")
    print("Results on the test data: \n")
    print("MAP = " + str(MAP))
    print("Precision at 500 = " + str(precision_at_500))

if __name__ == "__main__":

    # Create the spark session object
    conf = SparkConf()
    spark = SparkSession.builder.config(conf=conf).appName('tryout').getOrCreate()

    #spark = SparkSession.builder.appName('down_sample_tryout').getOrCreate()

    # Get the filename from the command line
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    rank = sys.argv[3]
    regParam = sys.argv[4]
    alpha = sys.argv[5]
    lower_bound = sys.argv[6]
    log_compression = sys.argv[7]


    # Call our main routine
    main(spark, train_data, test_data, rank, regParam, alpha, lower_bound, log_compression)
