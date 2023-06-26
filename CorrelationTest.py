import sys
import traceback

from pyspark.sql import functions as F
from matplotlib import pyplot as plt
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pandas.plotting import scatter_matrix
import pandas as pd
import six
from pyspark.sql.functions import from_unixtime, to_timestamp, month, dayofmonth, hour


def initialize():
    try:
        sc = SparkContext()
        sqlContext = SQLContext(sc)

        label = sys.argv[1]
        dataDf = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("data/readings.csv")

        dataDf.cache()
        dataDf.printSchema()
        pd.set_option('display.expand_frame_repr', False)
        return dataDf, label
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("Dataset Loading Error -> Cannot read the label")

def filterTimestamp(dataDf):
    dataDf = dataDf \
    .withColumn('timestamp', from_unixtime(dataDf['timestamp'] / 1000, 'yyyy-MM-dd HH:mm:ss')) \
    .withColumn('timestamp2', to_timestamp('timestamp'))

    dataDf = dataDf \
        .withColumn('month', month(dataDf['timestamp2'])) \
        .withColumn('day', dayofmonth(dataDf['timestamp2'])) \
        .withColumn('hour', hour(dataDf['timestamp2'])) \
        .drop(dataDf['timestamp']) \
        .drop(dataDf['timestamp2'])

    dataDf = dataDf \
        .withColumn('hour_range', F.floor(dataDf['hour'].cast("integer") / 3)) \
        .drop(dataDf['hour'])

    dataDf=dataDf.withColumn('hour_range', (dataDf['hour_range'].cast("integer")))
    return dataDf


def scatterPlot(dataDf, label):
    #scatter to see correlation
    dataDf.printSchema()
    numeric_features = [t[0] for t in dataDf.dtypes if t[1] == 'double' or t[1] == 'int' or t[1] == 'long']
    sampled_data = dataDf.select(numeric_features).sample(False, 0.8).toPandas()
    axs = scatter_matrix(sampled_data, figsize=(16, 16))
    n = len(sampled_data.columns)
    print(n)

    for i in dataDf.columns:
        if not( isinstance(dataDf.select(i).take(1)[0][0], six.string_types)):
            print( "Correlation to "+ label+" for ", i, dataDf.stat.corr(label,i))

    for i in range(n):
        v = axs[i, 0]
        v.yaxis.label.set_rotation(0)
        v.yaxis.label.set_ha('right')
        v.set_yticks(())
        h = axs[n-1, i]
        h.xaxis.label.set_rotation(90)
        h.set_xticks(())
    plt.show()


def main():
    dataDf, label=initialize()
    dataDf=filterTimestamp(dataDf)
    print(dataDf.toPandas().describe(include='all').transpose())
    scatterPlot(dataDf, label)

if __name__=="__main__":
    print("Correlation Test Starting")
    main()
    print("Correlation Test Finished")