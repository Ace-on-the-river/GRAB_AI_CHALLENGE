#!/home/solidsnake01/anaconda3/bin/python3.7

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("\nStarting script 1: Data Cleaning")
print("Starting script 1: Data Cleaning")
print("Starting script 1: Data Cleaning\n")

print("Warning: Some of the future steps in this pipeline required 32GB ram and might take many hours to run (depending on data amount). Best to run this on a good linux server.")
print("Warning: Some of the future steps in this pipeline required 32GB ram and might take many hours to run (depending on data amount). Best to run this on a good linux server.")
print("Warning: Some of the future steps in this pipeline required 32GB ram and might take many hours to run (depending on data amount). Best to run this on a good linux server.\n")

##########################################################################
########################  Remove duplicate labels ########################
##########################################################################

input_features_filename = "p3.csv"
label_filename = "labels.csv"

raw_label_df = pd.read_csv(label_filename, index_col="bookingID")

label_df = raw_label_df[~raw_label_df.index.duplicated(keep=False)] # remove dubplicated labels

print("Label shape before duplicate removal:", raw_label_df.shape)
print("Label shape after duplicate removal:", label_df.shape)

##########################################################################
########################  Remove duplicate labels ########################
##########################################################################



##########################################################################
###################  Remove duplicated feature entries ###################
##########################################################################

p1_raw = pd.read_csv(input_features_filename)
print("\nInput features shape:", p1_raw.shape)

p_raw = pd.concat([p1_raw], axis=0, ignore_index=True) # concat if multiple input files are provided

p_raw.sort_values(['bookingID', 'second'], inplace=True) # sort by booking then second

concated_raw_df = p_raw[p_raw['bookingID'].isin(label_df.index)] # remove duplicated labels

print("\nDebug:", concated_raw_df.shape, len(set(concated_raw_df['bookingID'])))
print("\nDebug:", len(set(label_df.index).intersection(set(concated_raw_df['bookingID']))))

##########################################################################
###################  Remove duplicated feature entries ###################
##########################################################################



##########################################################################
#############  Impute negative speed with surrounding values #############
##########################################################################

def conditional_impute_negative_speed_with_previous_value(data_df):
    bookingID_array = data_df['bookingID'].values
    speed_array = data_df['Speed'].values
    for index, element in enumerate(speed_array):
        entry_is_nan = np.isnan(np.array(speed_array[index]))
        if index == 0 and entry_is_nan:
            speed_array[index] = 0
        if index != 0 and entry_is_nan:
            if bookingID_array[index] == bookingID_array[index-1]:
                speed_array[index] = speed_array[index-1] # backward imputation
            if bookingID_array[index] == bookingID_array[index+1]:
                if np.isnan(np.array(speed_array[index+1])):
                    speed_array[index] = 0 # forward entry is NaN, impute with 0
                else:
                    speed_array[index] = speed_array[index+1] # forward imputation
            else:
                speed_array[index] = 0 # the sole data entry of bookingID on record
    return speed_array

# imputation is prefered over deleting that entry due to the possibility that other features might be useful for classification
concated_raw_df.loc[concated_raw_df['Speed'] < 0, "Speed"] = np.nan # negative speed are missing values, change them to NaN for imputation
concated_raw_df['Speed'] = conditional_impute_negative_speed_with_previous_value(concated_raw_df)

##########################################################################
#############  Impute negative speed with surrounding values #############
##########################################################################



##########################################################################
###################### Dubious second entry removal ######################
##########################################################################

print("\nInput feature shape before second removal:", concated_raw_df.shape)
concated_raw_df.loc[concated_raw_df['second'] > 3900, "second"] = np.nan # removed entries with abnormal second (greater than 7 hours)
concated_raw_df.dropna(inplace=True)
concated_raw_df.set_index("bookingID", drop=False, inplace=True)
print("Input feature shape after second removal: ", concated_raw_df.shape)

##########################################################################
###################### Dubious second entry removal ######################
##########################################################################



##########################################################################
#########################  Export to hdf5 format #########################
##########################################################################

if not os.path.exists("CLEANED_RAW_DATA"): os.mkdir("CLEANED_RAW_DATA")
concated_raw_df.to_hdf('CLEANED_RAW_DATA/CLEANED_RAW_DATA.h5', key='CLEANED_RAW_DATA_ALL', mode='w')
label_df.sort_index().to_hdf('CLEANED_RAW_DATA/CLEANED_RAW_DATA.h5', key='CLEANED_LABELS_ALL')

print("\nEnd of script 1: Clean data exported !")
print("End of script 1: Clean data exported !")
print("End of script 1: Clean data exported !\n")

##########################################################################
#########################  Export to hdf5 format #########################
##########################################################################