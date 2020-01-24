# --- Python Imports --- 
import csv
import random
import numpy as np
import os

# --- Sklearn Impports ---
from sklearn.ensemble import RandomForestRegressor

# --- Internal Imports ---
from unibw import R2
from unibw import loadCSVToDict

# ---------------------------------------------------
filename        = "csvdata/data_pressure.csv"
featureNames    = [ "W",
                    "L/D",
                    "theta",
                    "R"]
targetName      = "iso"
testFraction    = 0.2

# ---------------------------------------------------
# Load data
data    = loadCSVToDict(filename)

# ---------------------------------------------------
# Model division
numberOfRecords = len(data[featureNames[0]])
indices         = [index for index in range(numberOfRecords)]
random.shuffle(indices)

trainSetX = [ 
                [ data[featureName][index] for featureName in featureNames ]
                for index in indices[0:int((1.0-testFraction)*numberOfRecords)]
            ]
trainSetY = [ 
                data[targetName][index]
                for index in indices[0:int((1.0-testFraction)*numberOfRecords)]
            ]
testSetX =  [ 
                [ data[featureName][index] for featureName in featureNames ]
                for index in indices[0:int(testFraction*numberOfRecords)]
            ]
testSetY =  [ 
                data[targetName][index]
                for index in indices[0:int(testFraction*numberOfRecords)]
            ]

trainSetX   = np.asarray(trainSetX)
trainSetY   = np.asarray(trainSetY)
testSetX    = np.asarray(testSetX)
testSetY    = np.asarray(testSetY)

# ---------------------------------------------------
# Create and train model
model   = RandomForestRegressor(    max_depth=8, 
                                    random_state=0,
                                    n_estimators=100
                                    )
model.fit(trainSetX,trainSetY)

# ---------------------------------------------------
# Evaluate model
pY      = model.predict(testSetX)

'''
for data, prediction in zip(testSetY, pY):
    print( str(data) + "\t" + str(prediction) )
'''

r2 = R2(testSetY, pY)
print( "R2:\t" + str(r2) )

# ---------------------------------------------------
# Save model (if it's good enough)
import pickle
# Check accuracy of the currently saved tree
fileName        = "../models/random_forest_" + targetName + ".bin"
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
    x           = np.concatenate( (trainSetX, testSetX) )
    y           = np.concatenate( (trainSetY, testSetY) )
    pYOld       = modelOld.predict( x )
    pYNew       = model.predict( x )
    rSquaredOld = R2(y, pYOld)
    rSquaredNew = R2(y, pYNew)
    if rSquaredNew > rSquaredOld:
        print( "Old random forest with an R2 value of " + str(rSquaredOld) + " has been replaced with a new one (R2=" + str(rSquaredNew) + ")" )
        writeToFile = True
    else:
        print( "Old tree prevails (R2=" + str(rSquaredOld) + ")" )
    
    file.close()

if writeToFile:
    with open(fileName, 'wb') as file:
        pickle.dump(model, file)