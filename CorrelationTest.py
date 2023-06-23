import sys

from matplotlib import pyplot as plt
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pandas.plotting import scatter_matrix
import pandas as pd
import six

sc = SparkContext()
sqlContext = SQLContext(sc)

label = sys.argv[1]
data_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("data/readings.csv")

data_df.cache()
data_df.printSchema()
pd.set_option('display.expand_frame_repr', False)

print(data_df.toPandas().describe(include='all').transpose())

#scatter to see correlation
numeric_features = [t[0] for t in data_df.dtypes if t[1] == 'double' or t[1] == 'int']
sampled_data = data_df.select(numeric_features).sample(False, 0.8).toPandas()
axs = scatter_matrix(sampled_data, figsize=(10, 10))
n = len(sampled_data.columns)

for i in data_df.columns:
    if not( isinstance(data_df.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to "+ label+" for ", i, data_df.stat.corr(label,i))

for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
plt.show()



