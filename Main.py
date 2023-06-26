import subprocess
import json
import traceback
from kafka import KafkaConsumer
#import os

#os.environ['SPARK_HOME'] = 'C:\Program Files\spark-3.3.2-bin-hadoop3'
#os.environ['JAVA_HOME'] = 'C:\Program Files\Java\jdk1.8.0_212'

# os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages commons-logging:commons-logging:1.1.3,org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2,org.apache.kafka:kafka-clients:2.8.1 Main.py'


def setupKafka():
    try:
        consumer = KafkaConsumer('RaspTest',
                                 group_id=None,
                                 auto_offset_reset='latest',
                                 bootstrap_servers=['192.168.43.204:9094'],
                                 value_deserializer=lambda m: json.loads(m.decode('utf-8')))
        if consumer is not None:
            for message in consumer:
                if message is not None:
                    incomingReading = message.value
                    print(incomingReading)
                    settedLabel = setLabel(incomingReading)
                    return settedLabel
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("Kafka Connection Error -> Cannot establish connection to the broker's topic")


def setLabel(incomingReading): #
    try:
        #file=open("./data/test.json") #test
        #incomingReading=json.load(file) #test
        with open('./data/newReading.json', 'w') as f:
            json.dump(incomingReading, f)
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("Json Setting Error -> Cannot save the Json file")
    try:
        for key, value in incomingReading.items():
            if value == '' or value is None or value == "undefined":
                label = key
                return label
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("Label Setting Error -> Cannot extract the label from incoming reading")


def callRegressionTests(label):
    subprocess.call(['cmd', '/c', 'python CorrelationTest.py', label])
    subprocess.call(['cmd', '/c', 'python LinearRegression.py', label])
    subprocess.call(['cmd', '/c', 'python IsotonicRegression.py', label])
    subprocess.call(['cmd', '/c', 'python DecisionTreeRegression.py', label])
    subprocess.call(['cmd', '/c', 'python GBTRegression.py', label])
    subprocess.call(['cmd', '/c', 'python RandomForestRegression.py', label])


def main():
    while (True):
        settedLabel = setupKafka()
        #settedLabel =setLabel() #test
        print("Label upon which operate the prediction -> " + settedLabel)
        callRegressionTests(settedLabel)
        print("Reconnecting to Kafka topic for further incoming messages.")


if __name__ == "__main__":
    print("Starting Regression Analysis")
    main()
    print("Regression Analysis Ended")
