import sys
import traceback
from pathlib import Path
from pyspark import SparkContext
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.feature import StandardScaler
from pyspark.sql import SQLContext
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import from_unixtime, month, dayofmonth, hour, to_timestamp
from pyspark.sql import functions as F
from setuptools._vendor.ordered_set import OrderedSet


"""Receive the label from the main. This is also compared with a reference set containing all the columns of the dataset, 
    so that we can remove the one utilized as label and utilize the remaining as features"""
def setLabel():
    try:
        label = sys.argv[1]
        referenceString=OrderedSet(['timestamp', 'humidity', 'light', 'pm10', 'pm2_5', 'pressure', 'rain', 'temperature', 'wind_dir', 'wind_dir_degrees', 'wind_force', 'wind_speed'])
        originalCols= list(referenceString - OrderedSet(label))
        print("Features -> "+" ".join(originalCols))
        print("Label -> "+label)
        return label
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("Label Setting Error -> Cannot retrieve the label")


"""Initilize sparkContext and sqlContext and read the dataset from the provided file"""
def initialize():
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    sc.setLogLevel("OFF")
    try:
        dataDf = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("data/readings.csv")
        return dataDf
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("Dataset Loading Error -> Cannot read the label")


"""Convert the UNIX timestamp in a more yyyy-MM-dd HH:mm:ss format.
   Then, we extract from the converted dataset only month, day and hour, which are more relevant.
   In the end, we create a column containing a hour range, so we obtain 8 range of 3 hour"""
def filterTimestamp(dataDf):
    dataDf=dataDf \
        .withColumn('timestamp', from_unixtime(dataDf['timestamp']/1000,'yyyy-MM-dd HH:mm:ss')) \
        .withColumn('timestamp2', to_timestamp('timestamp'))

    dataDf=dataDf \
        .withColumn('month', month(dataDf['timestamp2'])) \
        .withColumn('day', dayofmonth(dataDf['timestamp2'])) \
        .withColumn('hour', hour(dataDf['timestamp2'])) \
        .drop(dataDf['timestamp']) \
        .drop(dataDf['timestamp2'])

    dataDf=dataDf \
        .withColumn('hour_range', F.floor(dataDf['hour'].cast("integer")/3)) \
        .drop(dataDf['hour'])
    return dataDf


"""We extract the features from the modified dataframe and cache it to speed up future access. """
def defineFeatures(dataDf):
    features = dataDf.columns
    print("New features -> "+" ".join(features))
    dataDf.cache()
    dataDf.printSchema()
    pd.set_option('display.expand_frame_repr', False)
    print(dataDf.toPandas().describe(include='all').transpose())
    return features


"""not needed because the regressor it's already efficient"""
#standard_scaler = StandardScaler(inputCols=features, outputCol="features")


"""We vectorize the dataset utilizing the features and split the created dataframe  in train and test portions"""
def vectorizeAndSplit(dataDf, features):
    vectorAssembler = VectorAssembler(inputCols=features, outputCol='features')
    vDataDf = vectorAssembler.transform(dataDf)

    splits = vDataDf.randomSplit([0.7, 0.3])
    trainDf = splits[0]
    testDf = splits[1]
    return trainDf, testDf


"""The Gradient Boost Tree Regression model is prepared, as well as an evaluator and the GridSearch operation utilizing a ParamGrid
   both to be used in the CrossValidator operation. 5 folds is the optimal tradeoff between speed of training and accuracy of the results."""
def gbtRegression(label):
    try:
        gbt = GBTRegressor(featuresCol = 'features', labelCol = label)

        gbtParamGrid = (ParamGridBuilder()
                .addGrid(gbt.maxDepth, [2, 5, 10])
                .addGrid(gbt.maxBins, [10, 20, 40])
                .addGrid(gbt.maxIter, [5, 10, 20])
                .build())

        gbtEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol=label, metricName="rmse")

        # Create 5-fold CrossValidator
        gbtcv = CrossValidator(estimator = gbt,
                               estimatorParamMaps = gbtParamGrid,
                               evaluator = gbtEvaluator,
                               numFolds = 5)
        return gbtcv, gbtEvaluator
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("Model Creation Error -> Cannot perform the creation of the model")


"""We check if exist already a Gradient Boost Tree Regression Model for this label:
   if the model does not exist, we perform the training operation on the cross validation model, then the best found is extracted and saved for later use;
   if the model does exist, we simply load the model"""
def trainOrLoad(label, gbtcv, trainDf):
    modelPath=Path("./TrainedModels/GBTRegressionBestModels/"+label)
    if not modelPath.exists():
        try:
            gbtModel = gbtcv.fit(trainDf)
            bestModel=gbtModel.bestModel
            print(bestModel)
            bestModel.write().overwrite().save("./TrainedModels/GBTRegressionBestModels/"+label)
            print("Model Saved")
            return bestModel
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            print("Model Training Error -> Cannot perform the training of the model")
    else:
        try:
            bestModel= GBTRegressionModel.load("./TrainedModels/GBTRegressionBestModels/"+label)
            print("Model Loaded")
            return bestModel
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            print("Model Creation Error -> Cannot perform the creation of the model")


"""We perform a prediction utilizing the best model obtained by the cross validation training and the test portion of the dataset. 
   In the end, an evaluation is performed on the resulting predictions"""
def predictAndEvaluate(bestModel, label, testDf, gbtEvaluator):
    try:
        gbtPredictions = bestModel.transform(testDf)
        gbtPredictions.select("prediction", label, "features").show(5)

        print('RMSE:', gbtEvaluator.evaluate(gbtPredictions))
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("Model Prediction Error -> Cannot perform the prediction on the model")



def main():
    dataDf = initialize()
    label = setLabel()
    filteredDf = filterTimestamp(dataDf)
    features = defineFeatures(filteredDf)
    trainSplit, testSplit = vectorizeAndSplit(filteredDf, features)
    gbtcv, gbtEvaluator = gbtRegression(label)
    bestModel = trainOrLoad(label, gbtcv, trainSplit)
    predictAndEvaluate(bestModel, label, testSplit, gbtEvaluator)

if __name__=="__main__":
    print("Gradient Boost Tree Regression Test Starting")
    main()
    print("Gradient Boost Tree Regression Test Finished")