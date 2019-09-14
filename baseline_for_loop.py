#Part 1: Basic recommender system

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

def main(spark, train_data, val_data, downsample=True, extension=None):
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
    val = spark.read.parquet(val_data)
    
    ### if down-sample: down-sample train data to random 0.1%
    if downsample:
        train = train.sample(False, 0.00001, seed = 0)
        #val = val.sample(False, 0.00001, seed = 0)

    if extension != None:
        if extension == "log": # log-compression
            train = train.withColumn("log_count", log("count")) # apply log-compression

        elif extension == "drop": # drop low counts
            lower_bound = train.approxQuantile("count", [0.1], 0.25) # treat the 0.1 quantile of count data as the lower bound
            train = train.filter(train["count"]>int(lower_bound[0])) # filter out count data rows lower than the lower bound

    ### transform dataframe columns: user_id, track_id from string to float and put them in the pipeline
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_indexed", handleInvalid='skip')
    track_indexer = StringIndexer(inputCol="track_id", outputCol="track_id_indexed", handleInvalid='skip')
    pipeline = Pipeline(stages=[user_indexer, track_indexer])
    indexing_model = pipeline.fit(train) #learn (return: pipeline model)

    ### transform the datasets and create the view 
    train = indexing_model.transform(train) # return a dataframe with new columns 
    train.createOrReplaceTempView("train")
    val = indexing_model.transform(val) # return a dataframe with new columns 
    val.createOrReplaceTempView("val")

    # group by user_id, aggregate track_id_indexed for train and val 
    val_groupby = spark.sql("select user_id_indexed, collect_list(track_id_indexed) track_id_indexed_collections from val group by user_id_indexed")
    val_groupby.createOrReplaceTempView("val_groupby")
    
    # Build the recommendation model using ALS on the training data
    rank = np.arange(4,10,2)
    regParam = np.linspace(0.01,0.2,3)
    alpha = np.linspace(0.5,2,3)
    paramGrid = list(itertools.product(rank, regParam, alpha))
    MAP_lst = [] # store MAP results
    precision_at_500_lst = [] # store precision at 500 results

    for combo in paramGrid:
        rank_, regParam_, alpha_ = combo

        # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
        if extension == "log":
            ratingCol = "log_count"
        else:
            ratingCol = "count"
        als = ALS(rank=rank_, regParam=regParam_, alpha=alpha_, implicitPrefs=True, userCol="user_id_indexed", itemCol="track_id_indexed", ratingCol=ratingCol, coldStartStrategy="drop")

        # Save the model
        model = als.fit(train) # fit the pipeline onto training data

        # get top 500 recommendations
        userRecs = model.recommendForAllUsers(500) # return: dataframe (columns: user_id_indexed, recommendations) 
                                                    # [('user_id_indexed', 'int'), ('recommendations', 'array<struct<track_id_indexed:int,rating:float>>')]
        userRecs = userRecs.select(userRecs.user_id_indexed, userRecs.recommendations.track_id_indexed.alias("pred_list")) # with track_id_indexed only, no track_id
        userRecs.createOrReplaceTempView("userRecs") # create temporary view
        
        combined_df = spark.sql('''select val_groupby.user_id_indexed user_id_indexed, userRecs.pred_list pred_list, 
        val_groupby.track_id_indexed_collections track_id_indexed_collections from userRecs inner join val_groupby on val_groupby.user_id_indexed = userRecs.user_id_indexed''') # combine dfs wrg to user_id_indexed

        # use ranking metrics for evaluations
        predLabelsTuple = combined_df.rdd.map(lambda r: (r.pred_list, r.track_id_indexed_collections)) # result: tuple
        metrics = RankingMetrics(predLabelsTuple)
        MAP = metrics.meanAveragePrecision
        precision_at_500 = metrics.precisionAt(500)
        MAP_lst.append(MAP) # store MAP for each config
        precision_at_500_lst.append(precision_at_500) # store precision at 500 for each config
        # print out validation evaluation result
        print("---------------------------------------")
        print("configs: \n")
        print("rank = " + str(rank_) + " , regParam = " + str(regParam_) + " , alpha = " + str(alpha_))
        print("\n")
        print("MAP = " + str(MAP))
        print("Precision at 500 = " + str(precision_at_500))

    min_index = MAP_lst.index(np.max(MAP_lst))
    rank_opt, regParam_opt, alpha_opt  = paramGrid[min_index]
    print("---------------------------------------")
    print("optimal configs: \n")
    print("rank = " + str(rank_opt) + " , regParam = " + str(regParam_opt) + " , alpha = " + str(alpha_opt))
    print("\n")
    print("MAP = " + str(np.max(MAP_lst)))
    print("Precision at 500 =" + str(precision_at_500_lst[min_index]))

if __name__ == "__main__":

    # Create the spark session object
    conf = SparkConf()
    spark = SparkSession.builder.config(conf=conf).appName('tryout').getOrCreate()

    # Get the filename from the command line
    train_data= sys.argv[1]
    val_data = sys.argv[2]
    downsample  = sys.argv[3]
    extension = sys.argv[4]

    # Call our main routine
    main(spark, train_data, val_data, downsample, extension)
