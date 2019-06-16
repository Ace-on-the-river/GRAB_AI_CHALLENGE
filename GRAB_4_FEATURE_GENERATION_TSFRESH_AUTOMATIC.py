#!/home/solidsnake01/anaconda3/bin/python3.7

import pandas as pd
import numpy as np
import time
import os
import warnings
warnings.filterwarnings('ignore')

print("\nStarting script 4: tsFresh Automatic Feature Generation")
print("Starting script 4: tsFresh Automatic Feature Generation")
print("Starting script 4: tsFresh Automatic Feature Generation\n")

print("\nDo pip install tsfresh if yet to do so")
print("Do pip install tsfresh if yet to do so")
print("Do pip install tsfresh if yet to do so\n")

print("Warning: tsFresh module requires pandas 0.23.4 version. Use pip install tsfresh to automatically downgrade pandas to 0.23.4")
print("Warning: tsFresh module requires pandas 0.23.4 version. Use pip install tsfresh to automatically downgrade pandas to 0.23.4")
print("Warning: tsFresh module requires pandas 0.23.4 version. Use pip install tsfresh to automatically downgrade pandas to 0.23.4\n")

##########################################################################
############################## Initialization ############################
##########################################################################

number_of_cores = 0 # cores

assert number_of_cores != 0, "Please specify number of cores in the script. Recommended to have at least 16 real computing cores."

##########################################################################
############################## Initialization ############################
##########################################################################



##########################################################################
##############  Extract tsFresh features for each bookingID ##############
##########################################################################

""" the reason why to store the generated data on hard disk is because if somehow the process is disrupted, we can continue where we had left off """

from tsfresh import extract_features

def generate_tsfresh():
    
    print("Generating tsFresh Features...\n")
    
    data_df = pd.read_hdf('CLEANED_RAW_DATA/CLEANED_RAW_DATA.h5', key='CLEANED_RAW_DATA_ALL')
    booking_ID_list = data_df['bookingID'].unique()
    ts_master_df = pd.DataFrame({})
    
    if not os.path.exists("TSFRESH_RAW_FEATURES"): os.mkdir("TSFRESH_RAW_FEATURES")

    for i, booking_id in enumerate(booking_ID_list):
        
        # if entry is already processed, skip that entry
        if not os.path.exists("TSFRESH_RAW_FEATURES/{}.h5".format(booking_id)):
            
            # for each bookingID, generate tsfresh automatic features one by one
            # features have to be generated for each bookingID one by one to avoid global features being present for the entire dataset
            # for future predictions we do not have access to those global features
            
            small_p_raw_df = data_df.loc[data_df['bookingID'] == booking_id, :]
            features_filtered_direct_df = extract_features(small_p_raw_df, column_id='bookingID', column_sort='second', n_jobs=number_of_cores, disable_progressbar=True)
            features_filtered_direct_df.to_hdf("TSFRESH_RAW_FEATURES/{}.h5".format(booking_id), key='TSFRESH_RAW_FEATURES', mode='w')
            
            if i % 50 == 0:
                print("tsfresh features generation: entry {} out of {} generated".format(i, len(booking_ID_list)))

##########################################################################
##############  Extract tsFresh features for each bookingID ##############
##########################################################################



##########################################################################
#######################  Combine generated features ######################
##########################################################################

def extract_generated_features():
    
    # combine generated features on long term storage to a single dataframe
    
    filename_list = os.listdir("TSFRESH_RAW_FEATURES")
    ts_master_df = pd.DataFrame({})
    
    if not os.path.exists("GENERATED_FEATURES"): os.mkdir("GENERATED_FEATURES")
        
    for i, filename in enumerate(filename_list):
        
        if i % 100 == 0:
            print("Entry {} out of {} concatenated to dataframe".format(i, len(filename_list)))
            
        features_filtered_direct_df = pd.read_hdf("TSFRESH_RAW_FEATURES/{}".format(filename))
        features_filtered_direct_df['BookingID'] = np.array([[filename.split(".")[0]]])
        
        if ts_master_df.shape[0] == 0:
            ts_master_df = features_filtered_direct_df
        else:
            ts_master_df = ts_master_df.append(features_filtered_direct_df, ignore_index=True)
            
    ts_master_df.set_index("BookingID", drop=True, inplace=True)
    ts_master_df.index = ts_master_df.index.astype('int64')
    ts_master_df.to_hdf('GENERATED_FEATURES/TSFRESH_FEATURES_ALL.h5', key='TSFRESH_FEATURES_ALL', mode='w')
    
##########################################################################
#######################  Combine generated features ######################
##########################################################################



generate_tsfresh()
extract_generated_features()

print("\nEnd of script 4: tsFresh Features Generated !")
print("End of script 4: tsFresh Features Generated !")
print("End of script 4: tsFresh Features Generated !\n")