# --- Python Imports ---
import numpy as np

# ---------------------------------------------------
def normalizeData( data, norms, exampleAxis=1 ):
    if exampleAxis is 0:
        data = np.transpose( data, (1,0) )

    for sampleIndex in range(len( data[0] )):
        for classIndex, norm in enumerate(norms):
            data[classIndex,sampleIndex] /= (2*norm)
            data[classIndex,sampleIndex] += 0.5

    if exampleAxis is 0:
        data = np.transpose( data, (1,0) )

    return data


def deNormalizeData( data, norms, exampleAxis=1 ):
    if exampleAxis is 0:
        data = np.transpose( data, (1,0) )

    for sampleIndex in range(len( data[0] )):
        for classIndex, norm in enumerate(norms):
            data[classIndex,sampleIndex] -= 0.5
            data[classIndex,sampleIndex] *= (2*norm)

    if exampleAxis is 0:
        data = np.transpose( data, (1,0) )
        
    return data