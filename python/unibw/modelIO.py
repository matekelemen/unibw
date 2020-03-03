# --- Python Imports ---
import os
import pickle

# ---------------------------------------------------
def saveModel(  fileName,
                model   ):
    with open( fileName, 'wb' ) as file:
        pickle.dump( model, file )



def loadModel( fileName ):
    # Try opening the file (binary format)
    file = open(fileName, 'rb')
    if os.stat(fileName).st_size is 0:
        raise RuntimeError( "File is empty!")
    
    # Load model
    model = pickle.load( file, encoding='binary' )

    return model