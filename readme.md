# UniBW <a name="unibw"></a>

Basic tools and demo scripts for reading data from csv files and training/evaluating sklearn models on them.


## Prerequisites <a name="prerequisites"></a>

You'll need access to python 3 and have it on your ```path``` ( add the python executable to your path on [windows](https://geek-university.com/python/add-python-to-the-windows-path/) or [linux](https://stackoverflow.com/a/3402176/12350793) ). I don't recommend using Anaconda, since keeping packages up-to-date for it is a nightmare, and sometimes you can run into weird conflicts with it.

Install necessary python packages in the terminal:
* ``` python3 -m pip install --upgrade pip numpy sklearn ``` for linux
* ``` python -m pip install --upgrade pip numpy sklearn ``` for windows


## Folder Structure <a name="folderStructure"></a>

* Trained models: ```[source]/models```
* Executable scripts: ```[source]/python```
* CSV data files: ```[source]/python/csvdata```
* Functions and objects for reading data and training models: ```[source]/python/unibw```


## Executable Scripts <a name="executables"></a>

If you want to train models that are supported by default, you can run ```[source]/python/sklearn_runmodels.py```. This script will train several instances of all implemented models on the given dataset, and save the best performing ones from each model type to ```[source]/models```. Details on how this happens will be discussed later.

If you wish to check the performance of the saved models, run ```[source]/python/sklearn_evaluatemodels.py```. This script finds all models that use the given data and prints their [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) (R2) calculated on **the entire dataset** ( not only on the test set, meaning that these values will be higher than expected ). 

All other scripts are works in progress.


## Input Data Format <a name="dataFormat"></a>

All input data must be located in ```[source]/python/csvdata``` as .csv files. Open up an existing file and check its format, you'll find that it has the following layout:
```
<column_0_name>,    <column_1_name>,    ...,    <column_d_name>
<column_0_data_0>,  <column_1_data_0>,  ...,    <column_d_data_0>
<column_0_data_1>,  <column_1_data_1>,  ...,    <column_d_data_1>
.
.
.
<column_0_data_n>,  <column_1_data_n>,  ...,    <column_d_data_n>
```
Every dataset must be in this format, containing all features and labels. Name field must contain strings and data fields must contain numbers, with no empty fields or outliers. All columns must have an **equal number of entries**, and column **names must be unique**.

Specific columns can be selected by their names in the script to be used as features or labels. This means that some columns might not be used during training. However, columns from different data files cannot be used during the execution of a script. If you wish to use columns from multiple files, merge the files together and make sure they satisfy the required data format specified above.


## Training Models for a New Dataset <a name="trainingOnNewDataset"></a>

First of all, copy your data file into ```[source]/python/csvdata``` and make sure it is in the [required format](#dataFormat). Then open ```[source]/python/sklearn_runmodels.py```, and you'll see the following lines:
```
fileName        = "csvdata/Pressure_Time_Curve_Data.csv"
featureNames    = [ "charge_mass",
                    "offset"]
labelNames      = [ "pso_spherical",
                    "pso_hemispherical" ]
```
The code above should be self explanatory but just to clarify:
* ```fileName``` is the full name of the data file you want to load (with relative path from the location of the script). This variable is a string and should point to an existing file.
* ```featureNames``` is a list of strings (even if there is only one name), and matches the names of the columns in the data file that should be features (inputs) of the model.
* ```labelNames``` is a list of strings (even if there is only one name), and matches the names of the columns in the data file that should be labels (outputs) of the model. Keep in mind that all currently implemented models support only one output. This means that every label you specify here will have a separate trained model associated with it.

After setting everything you need, you may run the script. Make sure that your data file has [valid format](#dataFormat), that ```fileName``` points to it, and that all entries in ```featureNames``` and ```labelNames``` are column names in it.


## Setting Model and Training Parameters <a name="setParams"></a>

Open ```[source]/python/sklearn_runmodel.py``` and find the following lines:
```
# Data partitioning
trainRatio          = 0.8

.
.
.

# Number of runs per model
numberOfRuns        = 10
```
* ```trainRatio``` specifies what fraction of all examples should be used for training. The data is first shuffled, then the specified number of examples (rounded down) is used as the training set, the rest is used as the test set. Note that shuffling is random and happens right before training, meaning that consecutively training models yield different results.
* ```numberOfRuns``` specifies how many models are trained for a model type, the best of which will be saved at the end. In total, <```numberOfRuns``` * number of label names> models will be trained for each implemented model type.

Model-specific parameters can be set later down the script. Let's take a look at linear regression as an example:
```
models["Linear"] = {
                    "model"             : LinearRegression,
                    "modelArguments"    : { "fit_intercept" : True,
                                            "normalize"     : True  }
                    }
```
All model types are collected in a ```dictionary``` names ```models```. The model type name used in ```models``` can be freely chosen (```"Linear"``` in this case), and will be used during printing results and naming the saved model. Each model type must have a ```"model"``` and a ```"modelArguments"``` entry.
* ```"model"``` is the model object, and should be imported from [sklearn](https://scikit-learn.org/stable/). Note that this is just a class, not an instance of it.
* ```"modelArguments"``` is a dictionary containing all keyword arguments you want to pass to the model. You can a look up the model on the *sklearn* website to see what you can pass to it ([example](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) for the linear model). Note that not all possible arguments are in ```"modelArguments"```, so feel free to expand it if you wish.


## Evaluating Trained Models

After the models have been trained, the best performing ones from each model type are saved to ```[source]/models```. To load and compute their *coefficient of determination* for **the entire dataset**, set the data file in```[source]/python/sklearn_evaluatemodels.py``` (the same way as [here](#trainingOnNewDataset)) and run it. The performance of all model types and all labels will be printed.

Currently, there is no way of retrieving the performance of the models on their respective test sets. However, you can still get them during training. Open ```[source]/python/sklearn_runmodels.py``` and find the following lines:
```
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
```
After executing this part, ```model``` will hold the best-performing model of the current model type, and ```bestValue``` its performance.