# --- Python Imports ---
import numpy as np

# ---------------------------------------------------

def R2( data, prediction ):
    # Recursively call R2 if the inputs are multidimensional
    if (type(data[0]) is list) or (type(data[0]) is np.ndarray):
        Y = np.transpose( np.array(data), (1,0) )
        F = np.transpose( np.array(prediction), (1,0) )
        return [ R2( y, f ) for y, f in zip(Y, F) ]

    # Init
    mean        = np.mean(data)
    enumerator  = 0.0
    denominator = 0.0

    # Compute square sum error and variance
    for y, f in zip(data, prediction):
        denominator += (y - mean) **2
        enumerator  += (f - y) **2
    
    #print( np.transpose( np.asarray( [data,prediction] ), (1,0)) )
    
    return 1 - enumerator/denominator


def testR2():
    f   = np.random.rand( 10 )
    r2  = R2(f,f)
    if np.abs( r2 - 1.0 ) > 1e-15:
        print( str(r2) )
        raise ValueError("R2 implementation error!")