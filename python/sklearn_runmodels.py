# --- Python Imports ---
import multiprocessing as mp
from functools import partial

# --- SKLearn Imports ---
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor

# --- Internal Imports ---
from unibw import loadCSVToDict
from unibw import runAndEvaluate
from unibw import saveModel

# ---------------------------------------------------
# Setup
fileName            = "csvdata/Pressure_Time_Curve_Data.csv"
featureNames        = [ "charge_mass",
                        "offset"]
labelNames          = [ "pso_spherical",
                        "pso_hemispherical" ]

# Data partitioning
trainRatio          = 0.8

# Misc
verbose             = True
saveModels          = False
printPredictions    = False

# Number of runs per model
numberOfRuns        = 10

# ---------------------------------------------------
if __name__ == "__main__":
    # Prevent windows from whining
    mp.freeze_support()
    
    # Set up the models and their arguments
    models = dict()

    models["Linear"]        = {
        "model"             : LinearRegression,
        "modelArguments"    : { "fit_intercept" : True,
                                "normalize"     : True  }
                            }

    models["Ridge"]         = {
        "model"             : Ridge,
        "modelArguments"    : { "fit_intercept" : True,
                                "normalize"     : True,
                                "alpha"         : 1.0  }
                            }

    models["SVR"]           = {
        "model"             : SVR,
        "modelArguments"    : { "kernel" : "poly",
                                "degree"     : 3    }
                            }

    models["KernelRidge"]   = {
        "model"             : KernelRidge,
        "modelArguments"    : { "kernel"    : "linear",
                                "degree"    : 3,
                                "alpha"     : 1.0,
                                "coef0"     : None  }
                            }

    models["DecisionTree"]  = {
        "model"             : DecisionTreeRegressor,
        "modelArguments"    : { "max_depth" : None }
                            }

    models["Bagging"]       = {
        "model"             : BaggingRegressor,
        "modelArguments"    : { "base_estimator"    : None,
                                "n_estimators"      : 100 }
                            }

    models["RandomForest"]  = {
        "model"             : RandomForestRegressor,
        "modelArguments"    : { "max_depth"         : None,
                                "n_estimators"      : 100,
                                "criterion"         : "mse",
                                "min_samples_split" : 2,
                                "min_samples_leaf"  : 1,
                                "max_features"      : "auto" }
                            }

    models["GradientBoost"] = {
        "model"             : GradientBoostingRegressor,
        "modelArguments"    : { "max_depth"         : 3,
                                "n_estimators"      : 100,
                                "criterion"         : "friedman_mse",
                                "loss"              : "ls",
                                "subsample"         : 1.0  }
                            }

    models["AdaBoost"]      = {
        "model"             : AdaBoostRegressor,
        "modelArguments"    : { "base_estimator"    : DecisionTreeRegressor(max_depth=None),
                                "n_estimators"      : 100,
                                "loss"              : "linear" }
                            }

    # ---------------------------------------------------
    # Set up multiprocessing
    pool        = mp.Pool( mp.cpu_count() )

    # Load data
    dataDict    = loadCSVToDict( fileName )

    # Extract dataset name
    dataName    = ( fileName.rfind('/'), fileName.rfind('.') )
    dataName    = fileName[dataName[0]+1:dataName[1]]

    # ---------------------------------------------------
    # Run models
    for labelName in labelNames:
        for modelName in models:
            # Set model name
            models[modelName]["modelName"] = modelName
            print(modelName)
            
            # Common arguments
            models[modelName]["dataDict"]           = dataDict
            models[modelName]["inputFileName"]      = fileName
            models[modelName]["featureNames"]       = featureNames
            models[modelName]["labelName"]          = labelName
            models[modelName]["trainRatio"]         = trainRatio
            models[modelName]["verbose"]            = verbose
            models[modelName]["save"]               = saveModels
            models[modelName]["printPredictions"]   = printPredictions
            
            # Train models
            results = [ pool.apply_async( partial(runAndEvaluate, **models[modelName]) ) for runIndex in range(numberOfRuns) ]

            # Find best model
            model = None
            bestValue = None
            for index, candidate in enumerate(results):
                candidate = candidate.get()
                if model is None:
                    model       = candidate["model"]
                    bestValue   = candidate["R2"]
                elif candidate["R2"] > bestValue:
                    model       = candidate["model"]
                    bestValue   = candidate["R2"]

            # Save best model
            fileName = "../models/" + dataName + "_" + modelName + "_" + labelName + ".bin"
            saveModel( fileName, model )

            # Print best result
            print( "Best " + modelName + " R2\t: " + str(bestValue) )

    # Terminate
    pool.close()