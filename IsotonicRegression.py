import sys
import traceback
from pathlib import Path
from pyspark import SparkContext
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.feature import StandardScaler
from pyspark.sql import SQLContext
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import IsotonicRegression, IsotonicRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import from_unixtime, month, dayofmonth, hour, to_timestamp, col
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType



"""Receive the label from the main."""
def setLabel():
    try:
        label = sys.argv[1]
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
        newReading = sqlContext.read.json('./data/newReading.json')
        newReading = newReading.withColumn(sys.argv[1], col(sys.argv[1]).cast(DoubleType()))
        nr = newReading.na.fill(value=0.0, subset=[sys.argv[1]])

        return dataDf, nr
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


"""We cache the df to speed up future access. """
def cacheDataframe(dataDf):
    dataDf.cache()
    dataDf.printSchema()
    pd.set_option('display.expand_frame_repr', False)
    print(dataDf.toPandas().describe(include='all').transpose())


"""not needed because the regressor it's already efficient"""
#standard_scaler = StandardScaler(inputCols=features, outputCol="features")


"""We vectorize the dataset utilizing the features and split the created dataframe  in train and test portions"""
def vectorizeDataframe(dataDf, features):
    vectorAssembler = VectorAssembler(inputCols=features, outputCol='features')
    dataDf = vectorAssembler.transform(dataDf)
    return dataDf

def splitDataframe(dataDf):
    splits = dataDf.randomSplit([0.7, 0.3])
    trainDf = splits[0]
    testDf = splits[1]
    return trainDf, testDf


"""The Isotonic Regression model is prepared, as well as an evaluator and the GridSearch operation utilizing a ParamGrid
   both to be used in the CrossValidator operation. 5 folds is the optimal tradeoff between speed of training and accuracy of the results."""
def isotonicRegression(label):
    try:
        ir = IsotonicRegression(featuresCol = 'features', labelCol = label )

        # Create ParamGrid for Cross Validation
        irParamGrid = (ParamGridBuilder()
                       .addGrid(ir.isotonic, [True, False])
                       .build())
        irEvaluator = RegressionEvaluator(predictionCol="prediction",labelCol=label, metricName="rmse")

        # Create 5-fold CrossValidator with 3 level parallelism
        ircv = CrossValidator(estimator = ir,
                              estimatorParamMaps = irParamGrid,
                              evaluator = irEvaluator,
                              numFolds = 5,
                              parallelism=3)
        return ircv, irEvaluator
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("Model Creation Error -> Cannot perform the creation of the model")


"""We check if exist already a Isotonic Regression Model for this label:
   if the model does not exist, we perform the training operation on the cross validation model, then the best found is extracted and saved for later use;
   if the model does exist, we simply load the model"""
def trainOrLoad(label, ircv, trainDf):
    modelPath=Path("./TrainedModels/IsotonicRegressionBestModels/"+label)
    if not modelPath.exists():
        try:
            irModel = ircv.fit(trainDf)
            bestModel=irModel.bestModel
            print(bestModel)
            bestModel.write().overwrite().save("./TrainedModels/IsotonicRegressionBestModels/"+label)
            print("Model Saved")
            return(bestModel)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            print("Model training Error -> Cannot perform the training of the model")
    else:
        try:
            bestModel= IsotonicRegressionModel.load("./TrainedModels/IsotonicRegressionBestModels/"+label)
            print("Model Loaded")
            return bestModel
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            print("Model Creation Error -> Cannot perform the creation of the model")


"""We perform a prediction utilizing the best model obtained by the cross validation training and the test portion of the dataset. 
   In the end, an evaluation is performed on the resulting predictions"""
def predictAndEvaluate(bestModel, label, testDf, irEvaluator, nrFilteredDf):
    try:
        irPredictions = bestModel.transform(testDf)
        irPredictions.select('prediction', label, 'features').show(5)
        nrPredictions = bestModel.transform(nrFilteredDf)
        nrPredictions.select("prediction", label, "features").show()
        print('RMSE:', irEvaluator.evaluate(irPredictions))
        print("Boundaries in increasing order: %s\n" % str(bestModel.boundaries))
        print("Predictions associated with the boundaries: %s\n" % str(bestModel.predictions))
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("Model Prediction Error -> Cannot perform the prediction on the model")



def main():
    dataDf, newReading = initialize()
    label = setLabel()
    filteredDf = filterTimestamp(dataDf)
    features=filteredDf.columns
    features.remove(label)
    cacheDataframe(filteredDf)
    nrFilteredDf = filterTimestamp(newReading)
    filteredDf=vectorizeDataframe(filteredDf, features)
    trainSplit, testSplit = splitDataframe(filteredDf)
    nrFilteredDf=vectorizeDataframe(nrFilteredDf, features)
    ircv, irEvaluator = isotonicRegression(label)
    bestModel = trainOrLoad(label, ircv, trainSplit)
    predictAndEvaluate(bestModel, label, testSplit, irEvaluator, nrFilteredDf)

if __name__=="__main__":
    print("Isotonic Regression Test Starting")
    main()
    print("Isotonic Regression Test Finished")