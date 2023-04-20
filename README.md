# Predicting empathy score using eye-tracker data
In this repository you will find the following:


1. test_rawdata --> Zip file with the raw data of 30 participants, performing gaze typing.
2. control_rawdata --> Zip file with the raw data of 30 participants, performing free viewing.
3. questionnaries --> Zip file with two questionnaries before and after the task performed by the participants. This data contains the empathy scores we are predicting

This three zip files were collected from the [EyeT4Empathy dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9719458/), and can be found [here](https://drive.google.com/drive/folders/1SlvDzPxx-vHP3nCmTyEXrUPao6pRYPcA?usp=share_link).

4. feature_extraction.py --> Python file that contains a series of functions that automatizes the feature extraction of both the control and test group of the dataset.
This functions are called in a jupyter notebook to create a matrix for each and store them in a csv file. The functions are the following:
  - select_group()--> Automatize choosing between test or control group when doing feature extraction
  - preprocess()----> Call the functions that perform the feature extraction of one recording
  - label()---------> Extract labels/target from questionnaries to append to dataframe
  - flatten()-------> Aditional function to flatten a list of list into a list (helpful in the creation of label)

5. 

6. regression_models.ipynb --> Jupyter Notebook were we call our matrix and perform several regression models and compare them to a dummy regressor to understand the performance of each. 
