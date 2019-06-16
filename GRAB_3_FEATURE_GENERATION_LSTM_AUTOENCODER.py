#!/home/solidsnake01/anaconda3/bin/python3.7

from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import time
import os
import warnings
warnings.filterwarnings('ignore')

print("\nStarting script 3: LSTM Autoencoder Feature Generation")
print("Starting script 3: LSTM Autoencoder Feature Generation")
print("Starting script 3: LSTM Autoencoder Feature Generation\n")

print("Warning: This step might take hours depending on how much input data and how many computing threads are deployed")
print("Warning: This step might take hours depending on how much input data and how many computing threads are deployed")
print("Warning: This step might take hours depending on how much input data and how many computing threads are deployed\n")

##########################################################################
##############  Generated LSTM autoencoder encoded features ##############
##########################################################################

""" the reason why to store the generated data on hard disk is because if somehow the process is disrupted, we can continue where we had left off """

def extract_ts_encoding():
    
    print("Generating LSTM Autoencoder Features...\n")
    
    LSTM_autoencoder= load_model("UTILITY/LSTM_autoencoder_model.21276-0.0017796.hdf5")
    encoder = Model(LSTM_autoencoder.input, LSTM_autoencoder.layers[1].output)
    
    data_df = pd.read_hdf('CLEANED_RAW_DATA/CLEANED_RAW_DATA.h5', key='CLEANED_RAW_DATA_ALL')
    booking_ID_list = data_df['bookingID'].unique()
    ts_master_df = pd.DataFrame({})
    
    if not os.path.exists("LSTM_RAW_FEATURES"): os.mkdir("LSTM_RAW_FEATURES")
        
    for i, booking_id in enumerate(booking_ID_list):
        
        if not os.path.exists("LSTM_RAW_FEATURES/{}.h5".format(booking_id)):
            
            small_p_raw_df = data_df.loc[data_df['bookingID'] == booking_id, :]
            
            try:
                # do robustscaling and value clipping here
                j = RobustScaler().fit(data_df.iloc[:, 1:].values)
                r = j.transform(small_p_raw_df.iloc[:, 1:].values)
                r[r > 3] = 3
                
                # the LSTM cell can take in short time series only
                # so we have to break up our long time series into small chunks of 4 timesteps
                master_input_list = []
                for index in range(4, r.shape[0]):
                    smalle_ts_r = r[index-4:index]
                    master_input_list.append(smalle_ts_r)

                input_sequence = np.array(master_input_list)
                prediction = encoder.predict(input_sequence)
                prediction_concat = np.concatenate((prediction.max(axis=0), prediction.mean(axis=0), np.median(prediction, axis=0), prediction.min(axis=0)))
                prediction_concat = prediction_concat.reshape(1, -1)
                
                if i % 50 == 0:
                    print("LSTM autoencoding: entry {} out of {} generated".format(i, len(booking_ID_list)))
                
                features_filtered_direct_df = pd.DataFrame(prediction_concat)
                features_filtered_direct_df.to_hdf('LSTM_RAW_FEATURES/{}.h5'.format(booking_id), key='LSTM_RAW_FEATURES', mode='w')
            except:
                features_filtered_direct_df = pd.DataFrame(np.zeros((1, 3072)))
                features_filtered_direct_df.to_hdf('LSTM_RAW_FEATURES/{}.h5'.format(booking_id), key='LSTM_RAW_FEATURES', mode='w')
                print("skip autoencoding")

##########################################################################
##############  Generated LSTM autoencoder encoded features ##############
##########################################################################



##########################################################################
#######################  Combine generated features ######################
##########################################################################

def extract_generated_features():
    
    # combine generated features on long term storage to a single dataframe
    
    filename_list = os.listdir("LSTM_RAW_FEATURES")
    ts_master_df = pd.DataFrame({})
    
    if not os.path.exists("GENERATED_FEATURES"): os.mkdir("GENERATED_FEATURES")
        
    for i, filename in enumerate(filename_list):
        
        if i % 100 == 0:
            print("Entry {} out of {} concatenated to dataframe".format(i, len(filename_list)))
            
        features_filtered_direct_df = pd.read_hdf("LSTM_RAW_FEATURES/{}".format(filename))
        features_filtered_direct_df['BookingID'] = np.array([[filename.split(".")[0]]])
        
        if ts_master_df.shape[0] == 0:
            ts_master_df = features_filtered_direct_df
        else:
            ts_master_df = ts_master_df.append(features_filtered_direct_df, ignore_index=True)
            
    ts_master_df.set_index("BookingID", drop=True, inplace=True)
    ts_master_df.index = ts_master_df.index.astype('int64')
    ts_master_df.to_hdf('GENERATED_FEATURES/LSTM_FEATURES_ALL.h5', key='LSTM_FEATURES_ALL', mode='w')

##########################################################################
#######################  Combine generated features ######################
##########################################################################



extract_ts_encoding()
extract_generated_features()

print("\nEnd of script 3: LSTM Autoencoder Features Generated !")
print("End of script 3: LSTM Autoencoder Features Generated !")
print("End of script 3: LSTM Autoencoder Features Generated !\n")

print("\nIMPORTANT - Install tsFresh now: pip install tsfresh")
print("IMPORTANT - Install tsFresh now: pip install tsfresh")
print("IMPORTANT - Install tsFresh now: pip install tsfresh")
print("IMPORTANT - Install tsFresh now: pip install tsfresh")
print("IMPORTANT - Install tsFresh now: pip install tsfresh")