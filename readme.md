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

First of all, copy your data file into ```[source]/python/csvdata``` and make sure it satisfies the [required format](#dataFormat)