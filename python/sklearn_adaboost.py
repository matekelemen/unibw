# --- Python Imports --- 
import csv
import random
import numpy as np
import os

# --- Sklearn Impports ---
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# --- Internal Imports ---
from unibw import R2
from unibw import loadCSVData, partitionDataSets

# ---------------------------------------------------
filename            = "csvdata/data_pressure.csv"
featureNames        = [ "W",
                        "L/D",
                        "theta",
                        "R"]
labelName           = "iso"
trainRatio          = 0.8

printPredictions    = False

# ---------------------------------------------------
# Load and divide data
features, labels                                        = loadCSVData(filename, featureNames, [labelName] )
trainFeatures, trainLabels, testFeatures, testLabels    = partitionDataSets( features, labels, trainRatio=trainRatio )

del features
del labels

# ---------------------------------------------------
# Create and train model
model   = AdaBoostRegressor(    base_estimator=DecisionTreeRegressor( max_depth=10 ),
                                n_estimators=100,
                                loss="square")
model.fit(trainFeatures,trainLabels)

# ---------------------------------------------------
# Evaluate model
pY      = model.predict(testFeatures)

if printPredictions:
    for data, prediction in zip(testLabels, pY):
        print( str(data) + "\t" + str(prediction) )

r2 = R2(testLabels, pY)
print( "R2:\t" + str(r2) )

# ---------------------------------------------------
# Save model (if it's good enough)
import pickle
# Check accuracy of the currently saved tree
fileName        = "../models/adaboost_" + labelName + ".bin"
file            = None
writeToFile     = False
try:
    file = open(fileName, 'rb')
    if os.stat(fileName).st_size is 0:
        writeToFile = True
        file.close()
        file        = None
except:
    writeToFile = True
if file is not None:
    modelOld    = pickle.load(file, encoding='binary')
    x           = np.concatenate( (trainFeatures, testFeatures) )
    y           = np.concatenate( (trainLabels, testLabels) )
    pYOld       = modelOld.predict( x )
    pYNew       = model.predict( x )
    rSquaredOld = R2(y, pYOld)
    rSquaredNew = R2(y, pYNew)
    if rSquaredNew > rSquaredOld:
        print( "Old model with an R2 value of " + str(rSquaredOld) + " has been replaced with a new one (R2=" + str(rSquaredNew) + ")" )
        writeToFile = True
    else:
        print( "Old model prevails (R2=" + str(rSquaredOld) + ")" )
    
    file.close()

if writeToFile:
    with open(fileName, 'wb') as file:
        pickle.dump(model, file)