from __future__ import absolute_import, division, print_function, unicode_literals

# --- Python Imports --- 
import csv
import random
import numpy as np
import os
from matplotlib import pyplot as plt

# --- Tensorflow Imports --- 
import tensorflow as tf

# --- Internal Imports ---
from unibw import R2
from unibw import loadCSVData, partitionDataSets
from unibw import normalizeData, deNormalizeData

# ---------------------------------------------------
# Data settings
fileName        = "csvdata/data_pressure.csv"
featureNames    = [ "W",
                    "L/D",
                    "theta",
                    "R"]
labelNames      = [ "iso",
                    "pso" ]
trainRatio      = 0.8
printTestSet    = True
showPlots       = True

# Model settings
numberOfNodes   = [ 200 for i in range(3) ]

# Optimization settings
numberOfEpochs  = 150
optimizer       = 'adam'

# ---------------------------------------------------
# Load data
features, labels    = loadCSVData(fileName, featureNames, labelNames )

# ---------------------------------------------------
# Normalize data
featureCoefficients = [np.max(np.abs(feature)) for feature in features]
labelCoefficients   = [np.max(np.abs(label)) for label in labels]

features    = normalizeData( features, featureCoefficients )
labels      = normalizeData( labels, labelCoefficients )

# Check data validity
def validData(lst):
    valid = True
    if np.any(np.isnan(lst)):
        valid = False
        raise ValueError("Nan in input data!")
    if np.max(np.abs( lst )) > 1.0:
        valid = False
        raise ValueError("Data out of bounds (>1.0)")
    if np.min( lst ) < 0.0:
        valid = False
        raise ValueError("Data out of bounds (<0.0)")

    return valid


validData(features)
validData(labels)

# Divide data
trainFeatures, trainLabels, testFeatures, testLabels    = partitionDataSets( features, labels, trainRatio=trainRatio )

# Release data memory
del features, labels

# ---------------------------------------------------
# Build model
model   = tf.keras.models.Sequential()
model.add( tf.keras.layers.InputLayer( input_shape=(len(featureNames)) ) )
for num in numberOfNodes:
    model.add( tf.keras.layers.Dense(num, activation='relu') )

model.add( tf.keras.layers.Dropout( 0.2 ) )
model.add( tf.keras.layers.Dense(len(labelNames)) )

# Set optimizer
model.compile(  optimizer=optimizer,
                loss='mean_squared_error')

# Display model structure
model.summary()

# Train network
model.fit(  trainFeatures, 
            trainLabels, 
            epochs=numberOfEpochs,
            shuffle=False,
            use_multiprocessing=True,
            workers=6)

# Evaluate network
print("\nEVALUATION")
model.evaluate(testFeatures,  testLabels, verbose=2)
print("\n")

# ---------------------------------------------------
# Save model
modelPath   = os.path.realpath(__file__)
modelPath   = modelPath[0:-modelPath[::-1].index("/")]
model.save(os.path.realpath( modelPath + "/../models/test_network.h5" ))

# Predict
y               = np.asarray( model.predict(testFeatures) )

# De-normalize
testFeatures    = deNormalizeData( testFeatures, featureCoefficients, exampleAxis=0 )
testLabels      = deNormalizeData( testLabels, labelCoefficients, exampleAxis=0 )
y               = deNormalizeData( y, labelCoefficients, exampleAxis=0 )

# Print predictions
if printTestSet:
    print("MODEL PREDICTIONS")
    for X, Y, pY in zip(testFeatures, testLabels, y):
        stringFormat = "%.4f"

        string = "["
        for value in X:
            string += stringFormat % value
            string += ","
        string = string[0:-1] + "]"

        string += "\t->"
        for value in pY:
            string += stringFormat % value
            string += ","
        string = string[0:-1]

        string += "\t("
        for value in Y:
            string += stringFormat % value
            string += ","
        string = string[0:-1] + ")"

        enumerator      = 0.0
        denominator     = 0.0
        for value, predicted in zip(Y, pY):
            enumerator  += (predicted - value)**2
            denominator += value
        string += "\t- error: "
        string += stringFormat % abs( np.sqrt(enumerator)/denominator )

        print( string )

# Compute R2
print( "R2 (train):\t" + str( R2(trainLabels, model.predict(trainFeatures) ) ) )
print( "R2 (test):\t" + str( R2(testLabels, y) ) + "\n" )

if showPlots:
    for targetIndex in range(len(labelNames)):
        for featureIndex in range(len(featureNames)):
            plt.subplot( len(featureNames), len(labelNames), targetIndex*len(featureNames) + featureIndex + 1 )
            plt.plot( [ testFeatures[index][featureIndex] for index in range(len(testFeatures)) ] , [testLabels[index][targetIndex] for index in range(len(testLabels))], 'b+' )
            plt.plot( [ testFeatures[index][featureIndex] for index in range(len(testFeatures)) ], [y[index][targetIndex] for index in range(len(y))], 'r.' )
plt.show()