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
def get_avg_sacc_speed(df, group):
    '''
    Return average saccade speed
    '''
    temp = df[df['Recording name'] == group]
    diff = temp['Saccade'].diff().values
    # changes from 0-1 (i.e., diff = 1) mean start of saccade; changes from 1->0 (i.e. diff = -1) mean end of saccade
    start_idx = np.where(diff == 1)[0]
    end_idx = np.where(diff == -1)[0]
    i = 0
    while start_idx[0] > end_idx[i]:
        # print(i, start_idx[0], end_idx[i])
        i += 1
    end_idx = end_idx[i:]
    speeds = []
    for start, end in zip(start_idx, end_idx):
        assert end > start
        speeds.append(np.nanmean(temp['Speed'].iloc[start:end+1].values))  # average speed/saccade
    return len(speeds), np.asarray(speeds).mean()  # number of saccades and average speed across saccades


def get_avg_fix_duration(df, group, srate=120):
    '''
    Return average fixation duration
    param df: Dataframe with original recording
    param group: name of recording session
    param srate: sampling rate of data (default in dataset is 120 Hz)
    '''
    temp = df[df['Recording name'] == group]
    diff = temp['Fixation'].diff().values
    # changes from 0-1 (i.e., diff = 1) indicate start of fixation; changes from 1->0 (i.e. diff = -1) indicate end of fixation
    start_idx = np.where(diff == 1)[0]
    end_idx = np.where(diff == -1)[0]
    i = 0
    while start_idx[0] > end_idx[i]:
        # print(i, start_idx[0], end_idx[i])
        i += 1
    end_idx = end_idx[i:]
    durations = []
    for start, end in zip(start_idx, end_idx):
        assert end > start
        durations.append((end-start+1)/srate)  # duration of fixation (number of rows/sampling rate)
    return len(durations), np.asarray(durations).mean()  # number of fixations and average duration across fixations


def get_unclassified_count(df, group, srate=120):
    temp = df[df['Recording name'] == group]
    diff = temp['Eye_movement_type_Unclassified'].diff().values
    # changes from 0-1 (i.e., diff = 1) indicate start of unclassified; changes from 1->0 (i.e. diff = -1) indicate end of unclassified
    start_idx = np.where(diff == 1)[0]
    end_idx = np.where(diff == -1)[0]
    i = 0
    while start_idx[0] > end_idx[i]:
        # print(i, start_idx[0], end_idx[i])
        i += 1
    end_idx = end_idx[i:]
    durations = []
    for start, end in zip(start_idx, end_idx):
        assert end > start
        durations.append((end - start + 1) / srate)  # duration of fixation (number of rows/sampling rate)
    return len(durations), np.asarray(durations).mean()  # number of fixations and average duration across fixations

# ---------------------------------------------------------------------------
def preprocess(path, fname):
    # ------------------------------------------
    #               READING FILE
    # ------------------------------------------
    # Read the .tsv file that contains the raw data of the participant
    df_table = pd.read_table(path + fname, sep='\t', low_memory=False)
    
    # Remove calibration points in recording
    startPoints = df_table[df_table['Event'] == 'ImageStimulusStart'].index.values.astype(int)
    endPoints = df_table[df_table['Event'] == 'ImageStimulusEnd'].index.values.astype(int)
    
    # Store only image stimulus
    df = pd.DataFrame()

    for i in range(len(startPoints)):
        start = startPoints[i]
        end = endPoints[i]
        trial = df_table.iloc[start:end+1]
        df = pd.concat([df, trial])

    # Select correctly the participant in loop
    partiName = int(file[13:-4])
    print('Participant #', partiName)


    # Features we are keeping
    df_col = ['Recording timestamp', 'Participant name', 'Recording name', 'Recording duration',
              'Pupil diameter left', 'Pupil diameter right', 'Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)',
              'Eye movement type', 'Gaze event duration', 'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)']
    # Remove unnecessary columns
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
    # Check that we're saving the right participant name (one is from the filename, the other is from the file
    assert part_name == partiName, "Participant numbers don't match! %d != %d" % (partiName, part_name)
    df_features['Participant name'] = df_features['Participant name'].replace(prev, part_name)

    # Label encoder for feature --> 'Eye movement type'
    df_features['Eye movement type'] = df_features['Eye movement type'].replace(("EyesNotFound", np.nan), "Unclassified")
    df_features = pd.get_dummies(df_features, prefix='Eye_movement_type', columns=['Eye movement type'])

    # Columns that need to be changed from object to float
    objColumns = ['Pupil diameter left', 'Pupil diameter right', 'Gaze point X (MCSnorm)',
                  'Gaze point Y (MCSnorm)', 'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)']
    # Change (commas) to (decimals) and convert object to float64
    for feature in objColumns:
        df_features[feature] = df_features[feature].str.replace(',', '.').astype(float)

    # Create distance and time columns for SPEED and ACC
    #df_features['Time'] = pd.to_timedelta(df_features['Recording timestamp'], unit='us')
    df_features['Time'] = pd.to_datetime(df_features['Recording timestamp']).astype(np.int64) / int(1e6)  # seconds
    df_features['Delta Time'] = df_features['Time'].diff()
    df_features['Position'] = np.sqrt(df_features['Gaze point X (MCSnorm)']**2 + df_features['Gaze point Y (MCSnorm)']**2)
    df_features['Speed'] = df_features['Position'].diff() / df_features['Delta Time']
    df_features['Acceleration'] = df_features['Speed'].diff() / df_features['Delta Time']

    # Create Average Fixation Speed Feature
    # Manuel: MISSING!!! I could use some help :)
    df_features['Fixation'] = df_features['Eye movement type'].replace(("EyesNotFound", np.nan), "Unclassified")
    mapFixation = {'Fixation': 1, 'Saccade': 0, 'Unclassified': 0, 'EyesNotFound': 0}

    df_features['Fixation'] = df_features['Fixation'].replace(mapFixation)
    df_features['Saccade'] = 0
    df_features['Saccade'] = [1 for i in df_features['Eye movement type'].values if i == 'Saccade']

    # ------------------------------------------------
    #  Group by recording and extracting new features
    # -----------------------------------------------
    grouped_data = df_features.groupby('Recording name')
    df_recordings = pd.DataFrame()

    for name, group in grouped_data:
        # Creation of features from big dataset
        recDur    = group['Recording duration'].unique().tolist()[0]
        gazeAvg   = group['Gaze event duration'].mean()

        num_fixations, avg_fix_duration = get_avg_fix_duration(df_features, name, srate=120)
        num_saccades, avg_sacc_speed = get_avg_sacc_speed(df_features, name)

        num_unclassified, avg_unclassified_duration = get_unclassified_count(df_features, name, srate=120)

        # Dictionary with features extracted
        feature_dict = {'Recording name'           : name,
                        'Participant name'         : partiName,
                        'Mean Pupil diameter left' : group['Pupil diameter left'].mean(),
                        'Std Pupil diameter left'  : group['Pupil diameter left'].std(),
                        'Min Pupil diamater left'  : group['Pupil diameter left'].min(),
                        'Max Pupil diamater left'  : group['Pupil diameter left'].max(),
                        'Mean Pupil diameter right': group['Pupil diameter right'].mean(),
                        'Std Pupil diameter right' : group['Pupil diameter right'].std(),
                        'Min Pupil diamater right' : group['Pupil diameter right'].min(),
                        'Max Pupil diamater right' : group['Pupil diameter right'].max(),
                        'Num. of Fixations'        : num_fixations,
                        'Num. of Saccades'         : num_saccades,
                        'Num. of Unclassified'     : num_unclassified,
                        'Recording duration (s)'      : (recDur/1000),
                        'Mean Gaze event duration (s)': (gazeAvg/1000),
                        'Mean Fixation point X'    : group['Fixation point X (MCSnorm)'].mean(),
                        'Std Fixation point X'     : group['Fixation point X (MCSnorm)'].std(),
                        'Mean Fixation point Y'    : group['Fixation point Y (MCSnorm)'].mean(),
                        'Std Fixation point Y'     : group['Fixation point Y (MCSnorm)'].std(),
                        'Mean Gaze point X'        : group['Gaze point X (MCSnorm)'].mean(),
                        'Std Gaze point X'         : group['Gaze point X (MCSnorm)'].std(),
                        'Mean Gaze point Y'        : group['Gaze point Y (MCSnorm)'].mean(),
                        'Std Gaze point Y'         : group['Gaze point Y (MCSnorm)'].std(),
                        'Speed'                    : group['Speed'].mean(),
                        'Acceleration'             : group['Acceleration'].mean(),
                        'Avg Saccade Speed'        : avg_sacc_speed,
                        'Avg Fix Duration'         : avg_fix_duration,
                        'Avg Unclassif Duration'   : avg_unclassified_duration,
                        'Empathy Score'            : 0
                        }


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
