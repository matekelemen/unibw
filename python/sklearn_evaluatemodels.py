# --- Python Imports --- 
import csv
import pickle
import numpy as np
import os

# --- Internal Imports ---
from unibw import R2
from unibw import loadCSVData, partitionDataSets

# ---------------------------------------------------
fileName            = "csvdata/bayer_3_4_6_7_8_10_11.csv"
featureNames        = [ "charge_mass",
                        "range",
                        "elevation"]
labelNames          = [ "iso", 
                        "pso" ]

# ---------------------------------------------------

# Extract dataset name
dataName    = ( fileName.rfind('/'), fileName.rfind('.') )
dataName    = fileName[dataName[0]+1:dataName[1]]

# ---------------------------------------------------
for labelName in labelNames:
    # Load and divide data
    features, labels       = loadCSVData(fileName, featureNames, [labelName] )
    features, labels, x, y = partitionDataSets( features, labels, trainRatio=1.0 )

    del x
    del y

    # ---------------------------------------------------
    # Collect model names
    modelPath       = "../models/"
    dataCriterion   = dataName + "_"
    nameCriterion   = "_" + labelName + ".bin"

    modelNames = [ fileName for fileName in os.listdir(modelPath) if os.path.isfile(os.path.join(modelPath,fileName)) and nameCriterion in fileName and dataCriterion in fileName ]

    # ---------------------------------------------------
    print( "\nLabel name: " + labelName )
    print( "Model Name\t\tR2 (entire set)" )
    print( "---------------------------------------" )
    # Loop through models and evaluate them
    for name in modelNames:
        path    = os.path.join( modelPath, name )
        r2      = None

        with open(path,"rb") as file:
            model       = pickle.load(file, encoding="binary")
            prediction  = model.predict(features)
            r2          = R2(labels, prediction)

        separator = "\t\t"
        if len(name) - len(nameCriterion) - len(dataCriterion) < 8:
            separator += "\t"

        print( name[len(dataCriterion):-len(nameCriterion)] + separator + "%.4f" % r2 )

    print("\n")