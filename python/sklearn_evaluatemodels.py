# --- Python Imports --- 
import csv
import pickle
import numpy as np
import os

# --- Internal Imports ---
from unibw import R2
from unibw import loadCSVData, partitionDataSets

# ---------------------------------------------------
filename            = "csvdata/Pressure_Time_Curve_Data.csv"
featureNames        = [ "charge_mass",
                        "offset"]
labelName           = "pso_spherical"

# ---------------------------------------------------
# Load and divide data
features, labels       = loadCSVData(filename, featureNames, [labelName] )
features, labels, x, y = partitionDataSets( features, labels, trainRatio=1.0 )

del x
del y

# ---------------------------------------------------
# Collect model names
modelPath       = "../models/"
nameCriterion   = "_" + labelName + ".bin"

modelNames = [ fileName for fileName in os.listdir(modelPath) if os.path.isfile(os.path.join(modelPath,fileName)) and labelName in fileName ]

# ---------------------------------------------------
print("\nModel Name\t\tR2 (entire set)")
print("---------------------------------------")
# Loop through models and evaluate them
for name in modelNames:
    path    = os.path.join( modelPath, name )
    r2      = None

    with open(path,"rb") as file:
        model       = pickle.load(file, encoding="binary")
        prediction  = model.predict(features)
        r2          = R2(labels, prediction)

    separator = "\t\t"
    if len(name) - len(nameCriterion) < 8:
        separator += "\t"

    print( name[:-len(nameCriterion)] + separator + "%.3f" % r2 )

print("\n")