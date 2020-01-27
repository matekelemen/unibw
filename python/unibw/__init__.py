from .metrics import R2, testR2
from .dataimport import loadCSVToDict, loadCSVData, partitionDataSets
from .normalization import normalizeData, deNormalizeData
from .runmodel import runAndEvaluate 

testR2()