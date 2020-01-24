from __future__ import absolute_import, division, print_function, unicode_literals

# --- Python Imports --- 
import random
import numpy as np
import os
from matplotlib import pyplot as plt

# --- Data Processing Imports ---
import csv
import pandas as pd

# --- Tensorflow Imports --- 
import tensorflow as tf

# --- Internal Imports ---
from unibw import R2

# ---------------------------------------------------
# Data settings
filename        = "csvdata/data_pressure.csv"
featureNames    = [ "W",
                    "L/D",
                    "theta",
                    "R"]
targetNames     = [ "iso" ]
testFraction    = 0.2


# ---------------------------------------------------
# Load data
data    = dict()
header  = []

with open(filename) as file:
    reader = csv.reader(file,delimiter=',')
    for lineNumber, line in enumerate(reader):

        if lineNumber is 0:
            # Parse header
            for title in line:
                data[title] = []
            header = tuple(data.keys())
        else:
            # Parse data
            for index, value in enumerate(line):
                try:
                    value = float(value)
                except:
                    pass
                data[header[index]].append(value)




# ---------------------------------------------------
# Model division
numberOfRecords = len(data[featureNames[0]])
indices         = [index for index in range(numberOfRecords)]
random.shuffle(indices)

trainSetX = [ 
                [ data[featureName][index] for featureName in featureNames ]
                for index in indices[0:int((1-testFraction)*numberOfRecords)]
            ]
trainSetY = [ 
                [ data[name][index] for name in targetNames ]
                for index in indices[0:int((1-testFraction)*numberOfRecords)]
            ]
testSetX =  [ 
                [ data[featureName][index] for featureName in featureNames ]
                for index in indices[0:int(testFraction*numberOfRecords)]
            ]
testSetY =  [ 
                [ data[name][index] for name in targetNames ]
                for index in indices[0:int(testFraction*numberOfRecords)]
            ]


# Check data validity
def validData(lst):
    valid = True
    if np.any(np.isnan(lst)):
        valid = False
        print("Nan in input data!")
    if np.max(np.abs( lst )) > 1.0:
        valid = False
        print("Data out of bounds (>1.0)")
    if np.min( lst ) < 0.0:
        valid = False
        print("Data out of bounds (<0.0)")

    return valid


validData(trainSetX)
validData(trainSetY)
validData(testSetX)
validData(testSetY)


# Release data memory
del data

# Enumerate examples
numberOfTrainingRecords     = len(trainSetX)
numberOfTestRecords         = len(testSetX)

# ---------------------------------------------------
# Format Data
def convertToPandas(inputData, inputNames):
    temp = dict()
    for componentIndex, componentName in enumerate(inputNames):
        temp[componentName] = [ inputData[index][componentIndex] for index in range(len(inputData)) ]
    temp = pd.DataFrame( data=temp )
    print(temp)
    return temp

trainSetX   = convertToPandas(trainSetX, featureNames)
trainSetY   = convertToPandas(trainSetY, targetNames)
testSetX    = convertToPandas(testSetX, featureNames)
testSetY    = convertToPandas(testSetY, targetNames)

# ---------------------------------------------------
# Model definition

# Create feature columns
featureColumns  = []
for featureName in featureNames:
    featureColumns.append( tf.feature_column.numeric_column(featureName, dtype=tf.float32) )

# Define input functions
def makeInputFunction( x, y, numberOfEpochs=None ):
    def inputFunction():
        dataSet     = tf.data.Dataset.from_tensor_slices((dict(x),y))
        dataSet.repeat(numberOfEpochs)
        dataSet     = dataSet.batch(len(x))
        return dataSet
    return inputFunction

trainingInputFunction   = makeInputFunction(trainSetX, trainSetY, numberOfEpochs=None )
evaluationInputFunction = makeInputFunction(testSetX, testSetY, numberOfEpochs=1 )

# ---------------------------------------------------
# Model Training
numberOfBatches = 1
model   = tf.estimator.BoostedTreesRegressor(   featureColumns,
                                                n_batches_per_layer=numberOfBatches,
                                                n_trees=100,
                                                max_depth=6,
                                                train_in_memory=False)
'''
model = tf.estimator.LinearRegressor(   featureColumns  )
'''

model.train(    trainingInputFunction, 
                max_steps=100)

print(model)

result = model.evaluate( input_fn=evaluationInputFunction )
print(pd.Series(result))


y = list(model.predict( input_fn=evaluationInputFunction ) )


for value in y:
    print(value["predictions"][0])
