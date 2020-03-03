# --- Python Imports ---
import numpy as np
import os
import pickle

# --- Sklearn Imports ---
from sklearn.linear_model import LinearRegression

# --- Internal Imports ---
from . import R2
from . import loadDataFromDict, loadCSVData, partitionDataSets
from . import saveModel, loadModel

# ---------------------------------------------------
def runAndEvaluate( model=LinearRegression,
                    dataDict=None,
                    modelName="LinearRegression",
                    modelArguments={"fit_intercept" : True},
                    inputFileName="csvdata/data_pressure.csv",
                    featureNames=["W","L/D","theta","R"],
                    labelName="iso",
                    trainRatio=0.8,
                    verbose=True,
                    save=True,
                    printPredictions=False ):

    # Load and divide data
    if dataDict is None:
        features, labels                                        = loadCSVData(inputFileName, featureNames, [labelName])
    else:
        features, labels = loadDataFromDict( dataDict, featureNames, [labelName] )
    trainFeatures, trainLabels, testFeatures, testLabels    = partitionDataSets( features, labels, trainRatio=trainRatio )

    del features
    del labels
    
    # Instantiate and train model
    model       = model(**modelArguments)
    model.fit( trainFeatures, trainLabels )

    # Evaluate model
    prediction  = model.predict(testFeatures)

    # Print results
    if printPredictions:
        for data, pY in zip(testLabels, prediction):
            print( str(data) + "\t" + str(pY) )

    if verbose:
        print("R2 on test set:\t" + str( R2(testLabels, prediction) ))

    # ---------------------------------------------------
    # Evaluate model
    rSquaredNew = None
    x           = np.concatenate( (trainFeatures, testFeatures) )
    y           = np.concatenate( (trainLabels, testLabels) )
    pYNew       = model.predict( x )
    rSquaredNew = R2(y, pYNew)

    # Save model (if it's better than the existing one)
    if save:
        fileName        = "../models/" + modelName + "_" + labelName + ".bin"
        file            = None
        writeToFile     = False
        try:
            file = loadModel( fileName )
        except:
            writeToFile = True

        if file is not None:
            modelOld    = pickle.load(file, encoding='binary')
            pYOld       = modelOld.predict( x )
            rSquaredOld = R2(y, pYOld)
            if rSquaredNew > rSquaredOld:
                print( "Old " + modelName + " with an R2 value of " + str(rSquaredOld) + " has been replaced with a new one (R2=" + str(rSquaredNew) + ")" )
                writeToFile = True
            else:
                print( "Old " + modelName + " prevails (R2=" + str(rSquaredOld) + ")" )
            
            file.close()

        if not file.closed:
            file.close()

        if writeToFile:
            saveModel(fileName, model)

    return {"R2":rSquaredNew, "model": model}