from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import time
import datetime


def ts(tm=None):
    if tm is None: 
        timestamp = time.time()
        return "[" + str(format(datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))) + "]"
    else: 
        return str(tm)

def wt(report, word): 
    timestamp_str = ts()
    report.write(timestamp_str + " ---- " + word + "\n")

def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

def get_dummy(df,categoricalCols,continuousCols,labelCol): 
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.sql.functions import col
    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)) for c in categoricalCols ]
    # default setting: dropLast=True
    encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol="{0}_encoded".format(indexer.getOutputCol())) for indexer in indexers ]
    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders] + continuousCols, outputCol="features")
    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    model=pipeline.fit(df)
    data = model.transform(df)
    data = data.withColumn('label',col(labelCol))
    return data.select('features','label')

def main(spark, report): 
    wt(report, "Start reading data. ")
    df = spark.read.format("com.databricks.spark.csv").options(header="true", inferschema="true").load(DATA_PATH)
    wt(report, "Get Dummy Variable")
    transformed_dummy = get_dummy(df, ["cell", "mark", "ideas"], ["avo", "cur"], "valid")
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=100).fit(transformed_dummy)
    data = featureIndexer.transform(transformed_dummy)

    # Training Phase
    rf = GBTRegressor()
    pipeline = Pipeline(stages=[featureIndexer, rf])
    result_model = []
    result_training_time = []
    result_mse = []
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse")

    for iteration in range(4): 
        (trainingData, testData) = data.randomSplit([0.6, 0.4])
        wt(report, "Start Training, iteration " + str(iteration))
        start_time = time.time()
        model = pipeline.fit(trainingData)
        end_time = time.time()
        wt(report, "Training Time - iter " + str(iteration) + " : " + ts(end_time - start_time))

        # Evaluation
        predictions = model.transform(testData)
        mse = evaluator.evaluate(predictions)

        # Update variables. 
        result_training_time.append(ts(end_time - start_time))
        result_model.append(model)
        result_mse.append(mse)

    wt(report, "STATISTICS REPORT: ")
    for iteration in range(4): 
        wt(report, " ---- Iteration: " + str(iteration))
        wt(report, " -------- Training Time: " + str(result_training_time[iteration]))
        wt(report, " -------- Result MSE: " + str(result_mse[iteration]))
    
    wt(report, "DONE")

        

if __name__ == "__main__": 
    spark = SparkSession.builder.appName("SPARK_ML").getOrCreate()
    DATA_PATH = "spark_data/ml_data.csv"
    with open("report.txt", "w") as report: 
        main(spark, report)
