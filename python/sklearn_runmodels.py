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
fileName            = "csvdata/Pressure_Time_Curve_Data.csv"
featureNames        = [ "charge_mass",
                        "offset"]
labelName           = "pso_spherical"

# Data partitioning
trainRatio          = 0.8

# Misc
verbose             = True
saveModels          = True
printPredictions    = False

# Number of runs per model
numberOfRuns        = 10

# ---------------------------------------------------
# Set up models and their arguments
models = dict()

models["Linear"]        = {
    "model"             : LinearRegression,
    "modelName"         : "Linear",
    "modelArguments"    : { "fit_intercept" : True,
                            "normalize"     : True  },
    "inputFileName"     : fileName,
    "featureNames"      : featureNames,
    "labelName"         : labelName,
    "trainRatio"        : trainRatio,
    "verbose"           : verbose,
    "save"              : saveModels,
    "printPredictions"  : printPredictions
}

models["Ridge"]         = {
    "model"             : Ridge,
    "modelName"         : "Ridge",
    "modelArguments"    : { "fit_intercept" : True,
                            "normalize"     : True,
                            "alpha"         : 1.0  },
    "inputFileName"     : fileName,
    "featureNames"      : featureNames,
    "labelName"         : labelName,
    "trainRatio"        : trainRatio,
    "verbose"           : verbose,
    "save"              : saveModels,
    "printPredictions"  : printPredictions
}

models["SVR"]           = {
    "model"             : SVR,
    "modelName"         : "SVR",
    "modelArguments"    : { "kernel" : "poly",
                            "degree"     : 3    },
    "inputFileName"     : fileName,
    "featureNames"      : featureNames,
    "labelName"         : labelName,
    "trainRatio"        : trainRatio,
    "verbose"           : verbose,
    "save"              : saveModels,
    "printPredictions"  : printPredictions
}

models["KernelRidge"]   = {
    "model"             : KernelRidge,
    "modelName"         : "KernelRidge",
    "modelArguments"    : { "kernel"    : "linear",
                            "degree"    : 3,
                            "alpha"     : 1.0,
                            "coef0"     : None  },
    "inputFileName"     : fileName,
    "featureNames"      : featureNames,
    "labelName"         : labelName,
    "trainRatio"        : trainRatio,
    "verbose"           : verbose,
    "save"              : saveModels,
    "printPredictions"  : printPredictions
}

models["DecisionTree"]  = {
    "model"             : DecisionTreeRegressor,
    "modelName"         : "DecisionTree",
    "modelArguments"    : { "max_depth" : None },
    "inputFileName"     : fileName,
    "featureNames"      : featureNames,
    "labelName"         : labelName,
    "trainRatio"        : trainRatio,
    "verbose"           : verbose,
    "save"              : saveModels,
    "printPredictions"  : printPredictions
}

models["Bagging"]       = {
    "model"             : BaggingRegressor,
    "modelName"         : "Bagging",
    "modelArguments"    : { "base_estimator"    : None,
                            "n_estimators"      : 100 },
    "inputFileName"     : fileName,
    "featureNames"      : featureNames,
    "labelName"         : labelName,
    "trainRatio"        : trainRatio,
    "verbose"           : verbose,
    "save"              : saveModels,
    "printPredictions"  : printPredictions
}

models["RandomForest"]  = {
    "model"             : RandomForestRegressor,
    "modelName"         : "RandomForest",
    "modelArguments"    : { "max_depth"         : None,
                            "n_estimators"      : 100,
                            "criterion"         : "mse",
                            "min_samples_split" : 2,
                            "min_samples_leaf"  : 1,
                            "max_features"      : "auto" },
    "inputFileName"     : fileName,
    "featureNames"      : featureNames,
    "labelName"         : labelName,
    "trainRatio"        : trainRatio,
    "verbose"           : verbose,
    "save"              : saveModels,
    "printPredictions"  : printPredictions
}

models["GradientBoost"] = {
    "model"             : GradientBoostingRegressor,
    "modelName"         : "GradientBoost",
    "modelArguments"    : { "max_depth"         : 3,
                            "n_estimators"      : 100,
                            "criterion"         : "friedman_mse",
                            "loss"              : "ls",
                            "subsample"         : 1.0  },
    "inputFileName"     : fileName,
    "featureNames"      : featureNames,
    "labelName"         : labelName,
    "trainRatio"        : trainRatio,
    "verbose"           : verbose,
    "save"              : saveModels,
    "printPredictions"  : printPredictions
}

models["AdaBoost"]      = {
    "model"             : AdaBoostRegressor,
    "modelName"         : "AdaBoost",
    "modelArguments"    : { "base_estimator"    : DecisionTreeRegressor(max_depth=None),
                            "n_estimators"      : 100,
                            "loss"              : "linear" },
    "inputFileName"     : fileName,
    "featureNames"      : featureNames,
    "labelName"         : labelName,
    "trainRatio"        : trainRatio,
    "verbose"           : verbose,
    "save"              : saveModels,
    "printPredictions"  : printPredictions
}

for modelName in models:
    for runIndex in range(numberOfRuns):
        runAndEvaluate(**models[modelName])