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


def partitionDataSets( features, labels, trainRatio=0.8 ):
    '''
    Argument types:
        - features      : numpy.ndarray (NxM)
        - labels        : numpy.ndarray (PxM)
        - trainRatio    : float

    Return types:
        - trainFeatures : numpy.ndarray (NxM1)
        - trainLabels   : numpy.ndarray (PxM1)
        - testFeatures  : numpy.ndarray (NxM2)
        - testLabels    : numpy.ndarray (PxM2)

    Randomly partitions the input data (features and labels) into a training set and a test set.
    The number of training examples will be floor(trainRatio * M), the rest will be in the test set.
    '''
    # Check if features/labels contain only one class
    singleFeature       = False
    singleLabel         = False

    shape               = features.shape
    if len(shape) is 1:
        features = np.asarray( [features] )
        singleFeature = True
    shape               = labels.shape
    if len(shape) is 1:
        labels = np.asarray( [labels] )
        singleLabel = True

    # Create shuffled index array
    numberOfExamples    = len(features[0])
    indices             = [index for index in range(len( numberOfExamples ))]
    random.shuffle(indices)

    # Partition indices
    boundaryIndex       = np.floor( numberOfExamples * trainRatio )
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
        trainFeatures   = trainFeatures[0]
        testFeatures    = testFeatures[0]
    
    if singleLabel:
        trainLabels     = trainLabels[0]
        testLabels      = testLabels[0]

    # Return data
    return trainFeatures, trainLabels, testFeatures, testLabels