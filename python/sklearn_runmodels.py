# --- SKLearn Imports ---
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor

# --- Internal Imports ---
from unibw import runAndEvaluate

# ---------------------------------------------------
# Setup
fileName            = "csvdata/data_pressure.csv"
featureNames        = [ "W",
                        "L/D",
                        "theta",
                        "R"]
labelName           = "pso"

# Data partitioning
trainRatio          = 0.8

# Misc
verbose             = True
saveModels          = True
printPredictions    = False

# Number of runs per model
numberOfRuns        = 10

# ---------------------------------------------------
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

for modelName in models:
    # Set model name
    models[modelName]["modelName"] = modelName
    # Common arguments
    models[modelName]["inputFileName"]      = fileName
    models[modelName]["featureNames"]       = featureNames
    models[modelName]["labelName"]          = labelName
    models[modelName]["trainRatio"]         = trainRatio
    models[modelName]["verbose"]            = verbose
    models[modelName]["save"]               = saveModels
    models[modelName]["printPredictions"]   = printPredictions
    # Train model
    for runIndex in range(numberOfRuns):
        runAndEvaluate(**models[modelName])