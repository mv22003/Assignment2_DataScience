## The main function is to preprocess the dataset from the Eye4TEmpathy paper,
## in this paper we have eye-tracker data and by extracting the raw data using
## this function your able to extract several important features per recording.

## This function has two parameters
## 1. path --> this path should contain all the recording, either test or control
## 2. file --> this is recommended to be given by a 'os.listdir' loop

## The output of this is a dataframe of size (n_recording, n_features)

# The libraries needed are:
import pandas as pd
import numpy as np
import os
# ---------------------------------------------------------------------------
def preprocess(path, file):
    # ------------------------------------------
    #               READING FILE
    # ------------------------------------------
    # Read the .tsv file that contains the raw data of the participant
    df_table = pd.read_table(path + file,sep='\t',low_memory=False)
    
    # Remove calibration points in recording
    startPoints = df_table[df_table['Event']=='ImageStimulusStart'].index.values.astype(int)
    endPoints = df_table[df_table['Event']=='ImageStimulusEnd'].index.values.astype(int)
    
    # Store only image stimulus
    df = pd.DataFrame()

    for i in range(len(startPoints)):
        start = startPoints[i]
        end = endPoints[i]

        trial = df_table.iloc[start:end+1]
        df = pd.concat([df,trial])

    # Select correctly the participant in loop
    partiName = int(file[13:-4])
    print('Participant #',partiName)


    # Features we are keeping
    df_col = ['Recording timestamp','Participant name',
              'Recording name','Recording duration',
              'Pupil diameter left','Pupil diameter right',
              'Gaze point X (MCSnorm)','Gaze point Y (MCSnorm)',
              'Eye movement type','Gaze event duration',
              'Fixation point X (MCSnorm)','Fixation point Y (MCSnorm)']
    
    # Feature seletion base of data given and purpose of model
    df_features = df[df_col]

    # ------------------------------------------
    #     Feature processing and correction
    # ------------------------------------------
    # Change Recording name to integer
    record_name = df_features['Recording name'].unique()
    for i in range(len(record_name)):
        df_features = df_features.replace(record_name[i], i)

    # Change Participant name to integer
    prev = df_features['Participant name'].unique().tolist()
    part_name = int(df_features['Participant name'].unique().tolist()[-1][13:15])
    df_features['Participant name'] = df_features['Participant name'].replace(prev, part_name)

    # Label encoder for feature --> 'Eye movement type'
    df_features['Eye movement type'] = df_features['Eye movement type'].replace(("EyesNotFound",np.nan), "Unclassified")
    df_features = pd.get_dummies(df_features, prefix='Eye_movement_type',columns=['Eye movement type'])

    # Columns that need to be changed from object to float
    objColumns = ['Pupil diameter left','Pupil diameter right','Gaze point X (MCSnorm)',
                  'Gaze point Y (MCSnorm)','Fixation point X (MCSnorm)','Fixation point Y (MCSnorm)']

    # Change (commas) to (decimals) and convert object to float64
    for feature in objColumns:
        df_features[feature] = df_features[feature].str.replace(',','.').astype(float)

    # ------------------------------------------------
    #  Group by recording and extracting new features
    # -----------------------------------------------
    grouped_data = df_features.groupby('Recording name')
    df_recordings = pd.DataFrame()

    for name, group in grouped_data:
        # Creation of features from big dataset
        pupL_avg  = group['Pupil diameter left'].mean()
        pupL_std  = group['Pupil diameter left'].std()
        pupR_avg  = group['Pupil diameter right'].mean()
        pupR_std  = group['Pupil diameter right'].std()
        numFix    = group['Eye_movement_type_Fixation'].tolist().count(1)
        numSac    = group['Eye_movement_type_Saccade'].tolist().count(1)
        numUnc    = group['Eye_movement_type_Unclassified'].tolist().count(1)
        recDur    = group['Recording duration'].unique().tolist()[0]
        gazeAvg   = group['Gaze event duration'].mean()
        meanfixX  = group['Fixation point X (MCSnorm)'].mean()
        stdFixX   = group['Fixation point X (MCSnorm)'].std()
        meanfixY  = group['Fixation point Y (MCSnorm)'].mean()
        stdFixY   = group['Fixation point Y (MCSnorm)'].std()
        meanGazeX = group['Gaze point X (MCSnorm)'].mean()
        stdGazeX  = group['Gaze point X (MCSnorm)'].std()
        meanGazeY = group['Gaze point Y (MCSnorm)'].mean()
        stdGazeY  = group['Gaze point Y (MCSnorm)'].std()

        # NEW FEATURES DIAMETER 5-10-2023   
        pupL_min = group['Pupil diameter left'].min()
        pupL_max = group['Pupil diameter left'].max()
        pupR_min = group['Pupil diameter right'].min()
        pupR_max = group['Pupil diameter right'].max()
        
        # NEW FEATURES SPEED AND ACC 5-10-2023 (WITH GAZE POINT)
        # speedX = group['Gaze point X (MCSnorm)']
        # speedY = group['Gaze point Y (MCSnorm)']
        # accelX = group
        # accelY = group


        # NEW FEATURES AVERAGE FIXATION SPEED

       



        # Dictionary with features extracted
        feature_dict = {'Recording name'           : name,
                        'Participant name'         : partiName,
                        'Mean Pupil diameter left' : pupL_avg,
                        'Std Pupil diameter left'  : pupL_std,
                        'Min Pupil diamater left'  : pupL_min,
                        'Max Pupil diamater left'  : pupL_max,
                        'Mean Pupil diameter right': pupR_avg,
                        'Std Pupil diameter right' : pupR_std,
                        'Min Pupil diamater right'  : pupR_min,
                        'Max Pupil diamater right'  : pupR_max,
                        'Num. of Fixations'        : numFix,
                        'Num. of Saccades'         : numSac,
                        'Num. of Unclassified'     : numUnc,
                        'Recording duration (s)'   : (recDur/1000),
                        'Mean Gaze event duration (s)': (gazeAvg/1000),
                        'Mean Fixation point X'    : meanfixX,
                        'Std Fixation point X'     : stdFixX,
                        'Mean Fixation point Y'    : meanfixY,
                        'Std Fixation point Y'     : stdFixY,
                        'Mean Gaze point X'        : meanGazeX,
                        'Std Gaze point X'         : stdGazeX,
                        'Mean Gaze point Y'        : meanGazeY,
                        'Std Gaze point Y'         : stdGazeY,
                        # 'Speed of Fixation X'      : speedX,
                        # 'Speed of Fixation X'      : speedY,
                        # 'Acceleration of Fixation X'      : accelX,
                        # 'Acceleration of Fixation X'      : accelY,
                        'Empathy Score'            : 0}


        # Append the features for this recording name to the feature dataframe
        df_recordings = df_recordings.append(feature_dict, ignore_index=True)

    # Set the recording name as the index
    df_recordings.set_index('Recording name', inplace=True)

    # If rows cointain nan values, we drop them because it means the recording was unsuccessful.
    df_recordings = df_recordings.dropna(axis=0) 

    return df_recordings
#---------------------------------------------------------------------------------------------------
## Function that becomes helpful for the appending of the target in the df
## Transforms a list of list into a single list, the parameter is the list of lists
def flatten(l):
    return [item for sublist in l for item in sublist]

# ------------------------------------------------------------------------
# This function automatizes the selection of the participant group
def select_group(grp):
    if grp == 'test':
        path = 'C:\\Users\\mverd\\Desktop\\IMD\\ESSEX\\TERM2\\Modules\\Data Science and Decision Making\\Assignment2_Final\\rawdata\\test\\'
        groupSelect = 1
    elif grp == 'control':
        path = 'C:\\Users\\mverd\\Desktop\\IMD\\ESSEX\\TERM2\\Modules\\Data Science and Decision Making\\Assignment2_Final\\rawdata\\control\\'
        groupSelect = 2
    return path, groupSelect

#---------------------------------------------------------------------------------
## This function reads both of the questionnaries and creates the label for the 
## model to predict 

## The parameters are the path we the questionaries are store, and the current 
## group selected (test or control)
def label(pathQ, groupSelect):
    # Read first and second questionnarie
    quest1 = pd.read_csv(pathQ + os.listdir(pathQ)[0], encoding= 'unicode_escape', low_memory=False)
    # quest2 = pd.read_csv(pathQ + os.listdir(pathQ)[1], encoding= 'unicode_escape', low_memory=False)

    # Extract labels to predict (Original Score, before experiment) 
    score = quest1.iloc[:,-2]

    # Need to store the correct indexes (odd or even)
    if groupSelect == 1:
        list_par = list(range(0,59,2))
    elif groupSelect == 2:
        list_par = list(range(1,60,2))

    # Assign only the desired values
    label = score[list_par]

    # Drop index so that we always have from 0-->29 
    label = label.reset_index(drop=True)

    return label
