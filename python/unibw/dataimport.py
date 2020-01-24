# --- Python Imports ---
import csv
import random
import numpy as np

# ---------------------------------------------------
def loadCSVToDict(fileName):
    '''
    Argument types:
        - fileName      : string
    
    Return types:
        - dataDict      : dictionary

    Reads a .csv of the following format:
        - First row contains the header (name of the data in the corresponding column) in string format
        - numeric data columns, number of columns match the number of elements in the header
        - data columns have equal length

    Returns the data columns as floats, organized in a dictionary
    (keys match the corresponding string elements in the header)
    '''
    data    = dict()
    header  = []

    with open(fileName) as file:
        reader = csv.reader(file,delimiter=',')
        for lineNumber, line in enumerate(reader):

            if lineNumber is 0:
                # Parse header
                for title in line:
                    data[title] = []
                header = tuple(data.keys())
            else:
                # Parse data
                for index, value in enumerate(line):
                    try:
                        value = float(value)
                    except:
                        pass
                    data[header[index]].append(value)

    # Convert to numpy array
    for name in header:
        data[name] = np.asarray( data[name], dtype=np.float64 )
    
    return data


def loadCSVData( fileName, featureNames, labelNames ):
    '''
    Argument types:
        - fileName      : string
        - featureNames  : list of strings (N)
        - labelNames    : list of strings (P)

    Return types:
        - features      : numpy.ndarray (NxM)
        - labels        : numpy.ndarray (PxN)
    '''
    # Read and divide data
    data        = loadCSVToDict(fileName)
    features    = [ data[name] for name in featureNames ]
    labels      = [ data[name] for name in labelNames ]

    # Convert to ndarrays
    features    = np.asarray( features )
    labels      = np.asarray( labels )

    return features, labels


def partitionDataSets( features, labels, trainRatio=0.8 ):
    '''
    Argument types:
        - features      : numpy.ndarray (NxM)
        - labels        : numpy.ndarray (PxM)
        - trainRatio    : float

    Return types:
        - trainFeatures : numpy.ndarray (M1xN)
        - trainLabels   : numpy.ndarray (M1xP)
        - testFeatures  : numpy.ndarray (M2xN)
        - testLabels    : numpy.ndarray (M2xP)

    Randomly partitions the input data (features and labels) into a training set and a test set.
    The number of training examples will be floor(trainRatio * M), the rest will be in the test set.
    '''
    # Check if features/labels contain only one class
    singleFeature       = False
    singleLabel         = False

    shape               = features.shape
    if len(shape) is 1 or shape[0] is 1 or shape[1] is 1:
        features = np.asarray( [np.ravel(features)] )
        singleFeature = True

    shape               = labels.shape
    if len(shape) is 1 or shape[0] is 1 or shape[1] is 1:
        labels = np.asarray( [np.ravel(labels)] )
        singleLabel = True

    # Create shuffled index array
    numberOfExamples    = len(features[0])
    indices             = [index for index in range( numberOfExamples )]
    random.shuffle(indices)

    # Partition indices
    boundaryIndex       = int(np.floor( numberOfExamples * trainRatio ))
    trainIndices        = indices[:boundaryIndex]
    testIndices         = indices[boundaryIndex:]

    # Partition data
    trainFeatures       = [ [vector[index] for index in trainIndices] for vector in features ]
    trainLabels         = [ [vector[index] for index in trainIndices] for vector in labels ]
    testFeatures        = [ [vector[index] for index in testIndices] for vector in features ]
    testLabels          = [ [vector[index] for index in testIndices] for vector in labels ]

    # Convert to ndarrays
    trainFeatures       = np.asarray( trainFeatures )
    trainLabels         = np.asarray( trainLabels )
    testFeatures        = np.asarray( testFeatures )
    testLabels          = np.asarray( testLabels )

    # Restore single classes if necessary
    if singleFeature:
        trainFeatures   = np.ravel(trainFeatures)
        testFeatures    = np.ravel(testFeatures)
    else:
        trainFeatures   = np.transpose( trainFeatures, (1,0) )
        testFeatures    = np.transpose( testFeatures, (1,0) )
    
    if singleLabel:
        trainLabels     = np.ravel(trainLabels)
        testLabels      = np.ravel(testLabels)
    else:
        trainLabels     = np.transpose( trainLabels, (1,0) )
        testLabels      = np.transpose( testLabels, (1,0) )

    # Return data
    return trainFeatures, trainLabels, testFeatures, testLabels