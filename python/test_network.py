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
from unibw import R2, loadCSVToDict

# ---------------------------------------------------
# Data settings
filename        = "csvdata/data_pressure.csv"
featureNames    = [ "W",
                    "L/D",
                    "theta",
                    "R"]
targetNames     = [ "iso",
                    "pso" ]
testFraction    = 0.2
printTestSet    = False
showPlots       = True

# Model settings
numberOfNodes   = [ 70 for i in range(10) ]

# Optimization settings
numberOfEpochs  = 150

# ---------------------------------------------------
# Load data
data    = loadCSVToDict( filename )

# ---------------------------------------------------
# Model division
numberOfRecords = len(data[featureNames[0]])
indices         = [index for index in range(numberOfRecords)]
random.shuffle(indices)


# Normalize data
featureCoefficients = [np.max(np.abs(data[featureName])) for featureName in featureNames]
targetCoefficients  = [np.max(np.abs(data[featureName])) for featureName in targetNames]

def normalizeData(inputData, norm):
    for index in range(len( inputData )):
        inputData[index] /= (2*norm)
        inputData[index] += 0.5
    return inputData

def deNormalizeData(inputData, norm):
    for index in range(len( inputData )):
        inputData[index] -= 0.5
        inputData[index] *= norm
    return inputData
    

for name, coefficient in zip(featureNames, featureCoefficients):
    data[name] = normalizeData(data[name], coefficient)

for name, coefficient in zip(targetNames, targetCoefficients):
    data[name] = normalizeData(data[name], coefficient)


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

# ---------------------------------------------------
# Build model
model   = tf.keras.models.Sequential()
model.add( tf.keras.layers.InputLayer( input_shape=(len(featureNames)) ) )
for num in numberOfNodes:
    model.add( tf.keras.layers.Dense(num, activation='relu') )
model.add( tf.keras.layers.Dense(len(targetNames)) )

# Set optimizer
model.compile(  optimizer='adam',
                loss='mse',
                metrics=['mse','mae'])

# Display model structure
model.summary()

# Train network
model.fit(trainSetX, trainSetY, epochs=numberOfEpochs)

# Evaluate network
print("\nEVALUATION")
model.evaluate(testSetX,  testSetY, verbose=2)
print("\n")

# ---------------------------------------------------
# Save model
modelPath   = os.path.realpath(__file__)
modelPath   = modelPath[0:-modelPath[::-1].index("/")]
model.save(os.path.realpath( modelPath + "/../models/test_network.h5" ))

# Predict
vSetX       = testSetX
vSetY       = testSetY
y           = model.predict(vSetX)

# De-normalize
for setIndex in range(len(vSetX)):
    for componentIndex, coefficient in enumerate(featureCoefficients):
        vSetX[setIndex][componentIndex] -= 0.5
        vSetX[setIndex][componentIndex] *= coefficient

for setIndex in range(len(vSetY)):
    for componentIndex, coefficient in enumerate(targetCoefficients):
        vSetY[setIndex][componentIndex] -= 0.5
        vSetY[setIndex][componentIndex] *= coefficient

        y[setIndex][componentIndex]     -= 0.5
        y[setIndex][componentIndex]     *= coefficient

# Print predictions
if printTestSet:
    print("MODEL PREDICTIONS")
    for X, Y, pY in zip(vSetX, vSetY, y):
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
print( "R2:\t" + str( R2(vSetY, y) ) + "\n" )

if showPlots:
    for targetIndex in range(len(targetNames)):
        for featureIndex in range(len(featureNames)):
            plt.subplot( len(featureNames), len(targetNames), targetIndex*len(featureNames) + featureIndex + 1 )
            plt.plot( [ vSetX[index][featureIndex] for index in range(len(vSetX)) ] , [vSetY[index][targetIndex] for index in range(len(vSetY))], 'b+' )
            plt.plot( [ vSetX[index][featureIndex] for index in range(len(vSetX)) ], [y[index][targetIndex] for index in range(len(y))], 'r.' )
plt.show()