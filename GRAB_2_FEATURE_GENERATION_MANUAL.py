#!/home/ubuntu/anaconda3/bin/python3.7

import pandas as pd
import numpy as np
import pickle
import time
import os
import warnings
warnings.filterwarnings('ignore')

print("\nStarting script 2: Manual Feature Generation")
print("Starting script 2: Manual Feature Generation")
print("Starting script 2: Manual Feature Generation\n")

print("Warning: This step took 20 minutes for 16m input data rows. Requires 32 GB ram.")
print("Warning: This step took 20 minutes for 16m input data rows. Requires 32 GB ram.")
print("Warning: This step took 20 minutes for 16m input data rows. Requires 32 GB ram.\n")

##########################################################################
##############  Generated LSTM autoencoder encoded features ##############
##########################################################################

def absolute(np_array):
    return np.abs(np_array)

def squared(np_array):
    return np_array**2

def log_absolute(np_array):
    return np.log(np.abs(np_array)+0.0001)

def log_squared(np_array):
    return np.log((np_array**2)+0.0001)

# calculate per second param change (acceleration, gyro reading, etc)
def generate_param_change(param_list, booking_ID_list, second_list, applied_function):
    param_change_list = [0]
    for i in range(1, len(booking_ID_list)):
        if booking_ID_list[i] != booking_ID_list[i-1]:
            param_change_list.append(0)
        else:
            param_change = applied_function(param_list[i] - param_list[i-1])
            second_change = second_list[i] - second_list[i-1]
            param_change_per_second = (param_change/second_change) if second_change > 0 else 0
            param_change_list.append(param_change_per_second)
    return param_change_list 

##########################################################################
##############  Generated LSTM autoencoder encoded features ##############
##########################################################################



##########################################################################
########################## FEATURE INTERACTION ###########################
##########################################################################

def generate_feature_interaction(raw_feat_df, feature_type):
    
    # feature interactions between different axes of acceleration and gyro

    # sum interactions
    raw_feat_df['{}_a_delta_xy_sum'.format(feature_type)] = raw_feat_df['{}_a_x_delta'.format(feature_type)] + raw_feat_df['{}_a_y_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_xy_sum'.format(feature_type)] = raw_feat_df['{}_g_x_delta'.format(feature_type)] + raw_feat_df['{}_g_y_delta'.format(feature_type)]
    raw_feat_df['{}_a_delta_yz_sum'.format(feature_type)] = raw_feat_df['{}_a_y_delta'.format(feature_type)] + raw_feat_df['{}_a_z_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_yz_sum'.format(feature_type)] = raw_feat_df['{}_g_y_delta'.format(feature_type)] + raw_feat_df['{}_g_z_delta'.format(feature_type)]
    raw_feat_df['{}_a_delta_xz_sum'.format(feature_type)] = raw_feat_df['{}_a_x_delta'.format(feature_type)] + raw_feat_df['{}_a_z_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_xz_sum'.format(feature_type)] = raw_feat_df['{}_g_x_delta'.format(feature_type)] + raw_feat_df['{}_g_z_delta'.format(feature_type)]
    raw_feat_df['{}_a_delta_xyz_sum'.format(feature_type)] = raw_feat_df['{}_a_x_delta'.format(feature_type)] + raw_feat_df['{}_a_y_delta'.format(feature_type)] + raw_feat_df['{}_a_z_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_xyz_sum'.format(feature_type)] = raw_feat_df['{}_g_x_delta'.format(feature_type)] + raw_feat_df['{}_g_y_delta'.format(feature_type)] + raw_feat_df['{}_g_z_delta'.format(feature_type)]

    # product interactions
    raw_feat_df['{}_a_delta_xy_mul'.format(feature_type)] = raw_feat_df['{}_a_x_delta'.format(feature_type)] * raw_feat_df['{}_a_y_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_xy_mul'.format(feature_type)] = raw_feat_df['{}_g_x_delta'.format(feature_type)] * raw_feat_df['{}_g_y_delta'.format(feature_type)]
    raw_feat_df['{}_a_delta_yz_mul'.format(feature_type)] = raw_feat_df['{}_a_y_delta'.format(feature_type)] * raw_feat_df['{}_a_z_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_yz_mul'.format(feature_type)] = raw_feat_df['{}_g_y_delta'.format(feature_type)] * raw_feat_df['{}_g_z_delta'.format(feature_type)]
    raw_feat_df['{}_a_delta_xz_mul'.format(feature_type)] = raw_feat_df['{}_a_x_delta'.format(feature_type)] * raw_feat_df['{}_a_z_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_xz_mul'.format(feature_type)] = raw_feat_df['{}_g_x_delta'.format(feature_type)] * raw_feat_df['{}_g_z_delta'.format(feature_type)]
    raw_feat_df['{}_a_delta_xyz_mul'.format(feature_type)] = raw_feat_df['{}_a_x_delta'.format(feature_type)] * raw_feat_df['{}_a_y_delta'.format(feature_type)] * raw_feat_df['{}_a_z_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_xyz_mul'.format(feature_type)] = raw_feat_df['{}_g_x_delta'.format(feature_type)] * raw_feat_df['{}_g_y_delta'.format(feature_type)] * raw_feat_df['{}_g_z_delta'.format(feature_type)]

    # mixed interactions between acceleration and gyro
    raw_feat_df['{}_ag_delta_xx_mul'.format(feature_type)] = raw_feat_df['{}_a_x_delta'.format(feature_type)] * raw_feat_df['{}_g_x_delta'.format(feature_type)]
    raw_feat_df['{}_ag_delta_yy_mul'.format(feature_type)] = raw_feat_df['{}_a_y_delta'.format(feature_type)] * raw_feat_df['{}_g_y_delta'.format(feature_type)]
    raw_feat_df['{}_ag_delta_zz_mul'.format(feature_type)] = raw_feat_df['{}_a_z_delta'.format(feature_type)] * raw_feat_df['{}_g_z_delta'.format(feature_type)]
    raw_feat_df['{}_ag_delta_xy_mul'.format(feature_type)] = raw_feat_df['{}_a_x_delta'.format(feature_type)] * raw_feat_df['{}_g_y_delta'.format(feature_type)]
    raw_feat_df['{}_ag_delta_xz_mul'.format(feature_type)] = raw_feat_df['{}_a_x_delta'.format(feature_type)] * raw_feat_df['{}_g_z_delta'.format(feature_type)]
    raw_feat_df['{}_ag_delta_yx_mul'.format(feature_type)] = raw_feat_df['{}_a_y_delta'.format(feature_type)] * raw_feat_df['{}_g_x_delta'.format(feature_type)]
    raw_feat_df['{}_ag_delta_yz_mul'.format(feature_type)] = raw_feat_df['{}_a_y_delta'.format(feature_type)] * raw_feat_df['{}_g_z_delta'.format(feature_type)]
    raw_feat_df['{}_ag_delta_zx_mul'.format(feature_type)] = raw_feat_df['{}_a_z_delta'.format(feature_type)] * raw_feat_df['{}_g_x_delta'.format(feature_type)]
    raw_feat_df['{}_ag_delta_zy_mul'.format(feature_type)] = raw_feat_df['{}_a_z_delta'.format(feature_type)] * raw_feat_df['{}_g_y_delta'.format(feature_type)]

    # ratio interactions
    raw_feat_df['{}_a_delta_xy_ratio'.format(feature_type)] = raw_feat_df['{}_a_x_delta'.format(feature_type)] / raw_feat_df['{}_a_y_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_xy_ratio'.format(feature_type)] = raw_feat_df['{}_g_x_delta'.format(feature_type)] / raw_feat_df['{}_g_y_delta'.format(feature_type)]
    raw_feat_df['{}_a_delta_yz_ratio'.format(feature_type)] = raw_feat_df['{}_a_y_delta'.format(feature_type)] / raw_feat_df['{}_a_z_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_yz_ratio'.format(feature_type)] = raw_feat_df['{}_g_y_delta'.format(feature_type)] / raw_feat_df['{}_g_z_delta'.format(feature_type)]
    raw_feat_df['{}_a_delta_xz_ratio'.format(feature_type)] = raw_feat_df['{}_a_x_delta'.format(feature_type)] / raw_feat_df['{}_a_z_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_xz_ratio'.format(feature_type)] = raw_feat_df['{}_g_x_delta'.format(feature_type)] / raw_feat_df['{}_g_z_delta'.format(feature_type)]

    raw_feat_df['{}_a_delta_xyz_ratio'.format(feature_type)] = raw_feat_df['{}_a_x_delta'.format(feature_type)] / raw_feat_df['{}_a_y_delta'.format(feature_type)] / raw_feat_df['{}_a_z_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_xyz_ratio'.format(feature_type)] = raw_feat_df['{}_g_x_delta'.format(feature_type)] / raw_feat_df['{}_g_y_delta'.format(feature_type)] / raw_feat_df['{}_g_z_delta'.format(feature_type)]
    raw_feat_df['{}_a_delta_yzx_ratio'.format(feature_type)] = raw_feat_df['{}_a_y_delta'.format(feature_type)] / raw_feat_df['{}_a_z_delta'.format(feature_type)] / raw_feat_df['{}_a_x_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_yzx_ratio'.format(feature_type)] = raw_feat_df['{}_g_y_delta'.format(feature_type)] / raw_feat_df['{}_g_z_delta'.format(feature_type)] / raw_feat_df['{}_g_x_delta'.format(feature_type)]
    raw_feat_df['{}_a_delta_zxy_ratio'.format(feature_type)] = raw_feat_df['{}_a_z_delta'.format(feature_type)] / raw_feat_df['{}_a_x_delta'.format(feature_type)] / raw_feat_df['{}_a_y_delta'.format(feature_type)]
    raw_feat_df['{}_g_delta_zxy_ratio'.format(feature_type)] = raw_feat_df['{}_g_z_delta'.format(feature_type)] / raw_feat_df['{}_g_x_delta'.format(feature_type)] / raw_feat_df['{}_g_y_delta'.format(feature_type)]
    
    return raw_feat_df

##########################################################################
########################## FEATURE INTERACTION ###########################
##########################################################################



##########################################################################
############### SINGLE FEATURE MATHEMATICAL MODIFICATION #################
##########################################################################

def get_column_quantile(data_func_df, xtile_bins, column_type):
    return pd.cut(data_func_df[column_type], bins=xtile_bins[column_type], labels=False, include_lowest=True)

def generate_TS_features(p_raw, booking_ID_list, second_list):
    
    # feature change generation: abs, **2, log(abs), log(**2)
    
    print("\nGenerating Single Features..")
    
    xtile_bins = pickle.load(open("UTILITY/xtile_bins.p", "rb"))
    
    # absolute of metric
    p_raw['abs_accuracy'] = np.abs(p_raw['Accuracy'])
    p_raw['abs_bearing'] = np.abs(p_raw['Bearing'])
    p_raw['abs_a_x'] = np.abs(p_raw['acceleration_x'])
    p_raw['abs_a_y'] = np.abs(p_raw['acceleration_y'])
    p_raw['abs_a_z'] = np.abs(p_raw['acceleration_z'])
    p_raw['abs_g_x'] = np.abs(p_raw['gyro_x'])
    p_raw['abs_g_y'] = np.abs(p_raw['gyro_y'])
    p_raw['abs_g_z'] = np.abs(p_raw['gyro_z'])
    p_raw['abs_speed'] = np.abs(p_raw['Speed'])
    
    # quantile of metric
    p_raw['q_accuracy'] = get_column_quantile(p_raw, xtile_bins, "Accuracy")
    p_raw['q_bearing'] = get_column_quantile(p_raw, xtile_bins, "Bearing")
    p_raw['q_a_x'] = get_column_quantile(p_raw, xtile_bins, "acceleration_x")
    p_raw['q_a_y'] = get_column_quantile(p_raw, xtile_bins, "acceleration_y")
    p_raw['q_a_z'] = get_column_quantile(p_raw, xtile_bins, "acceleration_z")
    p_raw['q_g_x'] = get_column_quantile(p_raw, xtile_bins, "gyro_x")
    p_raw['q_g_y'] = get_column_quantile(p_raw, xtile_bins, "gyro_y")
    p_raw['q_g_z'] = get_column_quantile(p_raw, xtile_bins, "gyro_z")
    p_raw['q_speed'] = get_column_quantile(p_raw, xtile_bins, "Speed")
    
    # log absolute of metric
    p_raw['log_abs_accuracy'] = np.log(np.abs(p_raw['Accuracy'])+0.0001)
    p_raw['log_abs_bearing'] = np.log(np.abs(p_raw['Bearing'])+0.0001)
    p_raw['log_abs_a_x'] = np.log(np.abs(p_raw['acceleration_x'])+0.0001)
    p_raw['log_abs_a_y'] = np.log(np.abs(p_raw['acceleration_y'])+0.0001)
    p_raw['log_abs_a_z'] = np.log(np.abs(p_raw['acceleration_z'])+0.0001)
    p_raw['log_abs_g_x'] = np.log(np.abs(p_raw['gyro_x'])+0.0001)
    p_raw['log_abs_g_y'] = np.log(np.abs(p_raw['gyro_y'])+0.0001)
    p_raw['log_abs_g_z'] = np.log(np.abs(p_raw['gyro_z'])+0.0001)
    p_raw['log_abs_speed'] = np.log(np.abs(p_raw['Speed'])+0.0001)
    
    # square of metric
    p_raw['sqr_accuracy'] = (p_raw['Accuracy'])**2
    p_raw['sqr_bearing'] = (p_raw['Bearing'])**2
    p_raw['sqr_a_x'] = (p_raw['acceleration_x'])**2
    p_raw['sqr_a_y'] = (p_raw['acceleration_y'])**2
    p_raw['sqr_a_z'] = (p_raw['acceleration_z'])**2
    p_raw['sqr_g_x'] = (p_raw['gyro_x'])**2
    p_raw['sqr_g_y'] = (p_raw['gyro_y'])**2
    p_raw['sqr_g_z'] = (p_raw['gyro_z'])**2
    p_raw['sqr_speed'] = (p_raw['Speed'])**2
    
    # log square of metric
    p_raw['log_sqr_accuracy'] = np.log((p_raw['Accuracy']**2)+0.0001)
    p_raw['log_sqr_bearing'] = np.log((p_raw['Bearing']**2)+0.0001)
    p_raw['log_sqr_a_x'] = np.log((p_raw['acceleration_x']**2)+0.0001)
    p_raw['log_sqr_a_y'] = np.log((p_raw['acceleration_y']**2)+0.0001)
    p_raw['log_sqr_a_z'] = np.log((p_raw['acceleration_z']**2)+0.0001)
    p_raw['log_sqr_g_x'] = np.log((p_raw['gyro_x']**2)+0.0001)
    p_raw['log_sqr_g_y'] = np.log((p_raw['gyro_y']**2)+0.0001)
    p_raw['log_sqr_g_z'] = np.log((p_raw['gyro_z']**2)+0.0001)
    p_raw['log_sqr_speed'] = np.log((p_raw['Speed']**2)+0.0001)

    # per second change of absolute metric
    p_raw['abs_accuracy_delta'] = generate_param_change(p_raw['Accuracy'].values.tolist(), booking_ID_list, second_list, absolute)
    p_raw['abs_bearing_delta'] = generate_param_change(p_raw['Bearing'].values.tolist(), booking_ID_list, second_list, absolute)
    p_raw['abs_a_x_delta'] = generate_param_change(p_raw['acceleration_x'].values.tolist(), booking_ID_list, second_list, absolute)
    p_raw['abs_a_y_delta'] = generate_param_change(p_raw['acceleration_y'].values.tolist(), booking_ID_list, second_list, absolute)
    p_raw['abs_a_z_delta'] = generate_param_change(p_raw['acceleration_z'].values.tolist(), booking_ID_list, second_list, absolute)
    p_raw['abs_g_x_delta'] = generate_param_change(p_raw['gyro_x'].values.tolist(), booking_ID_list, second_list, absolute)
    p_raw['abs_g_y_delta'] = generate_param_change(p_raw['gyro_y'].values.tolist(), booking_ID_list, second_list, absolute)
    p_raw['abs_g_z_delta'] = generate_param_change(p_raw['gyro_z'].values.tolist(), booking_ID_list, second_list, absolute)
    p_raw['abs_speed_delta'] = generate_param_change(p_raw['Speed'].values.tolist(), booking_ID_list, second_list, absolute)

    # per second change of log absolute metric
    p_raw['log_abs_accuracy_delta'] = generate_param_change(p_raw['Accuracy'].values.tolist(), booking_ID_list, second_list, log_absolute)
    p_raw['log_abs_bearing_delta'] = generate_param_change(p_raw['Bearing'].values.tolist(), booking_ID_list, second_list, log_absolute)
    p_raw['log_abs_a_x_delta'] = generate_param_change(p_raw['acceleration_x'].values.tolist(), booking_ID_list, second_list, log_absolute)
    p_raw['log_abs_a_y_delta'] = generate_param_change(p_raw['acceleration_y'].values.tolist(), booking_ID_list, second_list, log_absolute)
    p_raw['log_abs_a_z_delta'] = generate_param_change(p_raw['acceleration_z'].values.tolist(), booking_ID_list, second_list, log_absolute)
    p_raw['log_abs_g_x_delta'] = generate_param_change(p_raw['gyro_x'].values.tolist(), booking_ID_list, second_list, log_absolute)
    p_raw['log_abs_g_y_delta'] = generate_param_change(p_raw['gyro_y'].values.tolist(), booking_ID_list, second_list, log_absolute)
    p_raw['log_abs_g_z_delta'] = generate_param_change(p_raw['gyro_z'].values.tolist(), booking_ID_list, second_list, log_absolute)
    p_raw['log_abs_speed_delta'] = generate_param_change(p_raw['Speed'].values.tolist(), booking_ID_list, second_list, log_absolute)

    # per second change of squared metric
    p_raw['sqr_accuracy_delta'] = generate_param_change(p_raw['Accuracy'].values.tolist(), booking_ID_list, second_list, squared)
    p_raw['sqr_bearing_delta'] = generate_param_change(p_raw['Bearing'].values.tolist(), booking_ID_list, second_list, squared)
    p_raw['sqr_a_x_delta'] = generate_param_change(p_raw['acceleration_x'].values.tolist(), booking_ID_list, second_list, squared)
    p_raw['sqr_a_y_delta'] = generate_param_change(p_raw['acceleration_y'].values.tolist(), booking_ID_list, second_list, squared)
    p_raw['sqr_a_z_delta'] = generate_param_change(p_raw['acceleration_z'].values.tolist(), booking_ID_list, second_list, squared)
    p_raw['sqr_g_x_delta'] = generate_param_change(p_raw['gyro_x'].values.tolist(), booking_ID_list, second_list, squared)
    p_raw['sqr_g_y_delta'] = generate_param_change(p_raw['gyro_y'].values.tolist(), booking_ID_list, second_list, squared)
    p_raw['sqr_g_z_delta'] = generate_param_change(p_raw['gyro_z'].values.tolist(), booking_ID_list, second_list, squared)
    p_raw['sqr_speed_delta'] = generate_param_change(p_raw['Speed'].values.tolist(), booking_ID_list, second_list, squared)

    # per second change of log squared metric
    p_raw['log_sqr_accuracy_delta'] = generate_param_change(p_raw['Accuracy'].values.tolist(), booking_ID_list, second_list, log_squared)
    p_raw['log_sqr_bearing_delta'] = generate_param_change(p_raw['Bearing'].values.tolist(), booking_ID_list, second_list, log_squared)
    p_raw['log_sqr_a_x_delta'] = generate_param_change(p_raw['acceleration_x'].values.tolist(), booking_ID_list, second_list, log_squared)
    p_raw['log_sqr_a_y_delta'] = generate_param_change(p_raw['acceleration_y'].values.tolist(), booking_ID_list, second_list, log_squared)
    p_raw['log_sqr_a_z_delta'] = generate_param_change(p_raw['acceleration_z'].values.tolist(), booking_ID_list, second_list, log_squared)
    p_raw['log_sqr_g_x_delta'] = generate_param_change(p_raw['gyro_x'].values.tolist(), booking_ID_list, second_list, log_squared)
    p_raw['log_sqr_g_y_delta'] = generate_param_change(p_raw['gyro_y'].values.tolist(), booking_ID_list, second_list, log_squared)
    p_raw['log_sqr_g_z_delta'] = generate_param_change(p_raw['gyro_z'].values.tolist(), booking_ID_list, second_list, log_squared)
    p_raw['log_sqr_speed_delta'] = generate_param_change(p_raw['Speed'].values.tolist(), booking_ID_list, second_list, log_squared)
    
    print("\nGenerating Feature Interactions..") # feature interactions
    p_raw = generate_feature_interaction(p_raw, "abs")
    p_raw = generate_feature_interaction(p_raw, "log_abs")
    p_raw = generate_feature_interaction(p_raw, "sqr")
    p_raw = generate_feature_interaction(p_raw, "log_sqr")
    
    return p_raw

##########################################################################
############### SINGLE FEATURE MATHEMATICAL MODIFICATION #################
##########################################################################



####################################################################################
########################## GROUPBY COMPRESSION FUNCTIONS ###########################
####################################################################################

def get_metric_attributes(raw_feat_df, groupby_df, feature_type, groupby_type):
    
    raw_feat_df['{}_accuracy_{}'.format(feature_type, groupby_type)] = groupby_df['{}_accuracy'.format(feature_type)]
    raw_feat_df['{}_bearing_{}'.format(feature_type, groupby_type)] = groupby_df['{}_bearing'.format(feature_type)]
    raw_feat_df['{}_a_x_{}'.format(feature_type, groupby_type)] = groupby_df['{}_a_x'.format(feature_type)]
    raw_feat_df['{}_a_y_{}'.format(feature_type, groupby_type)] = groupby_df['{}_a_y'.format(feature_type)]
    raw_feat_df['{}_a_z_{}'.format(feature_type, groupby_type)] = groupby_df['{}_a_z'.format(feature_type)]
    raw_feat_df['{}_g_x_{}'.format(feature_type, groupby_type)] = groupby_df['{}_g_x'.format(feature_type)]
    raw_feat_df['{}_g_y_{}'.format(feature_type, groupby_type)] = groupby_df['{}_g_y'.format(feature_type)]
    raw_feat_df['{}_g_z_{}'.format(feature_type, groupby_type)] = groupby_df['{}_g_z'.format(feature_type)]
    raw_feat_df['{}_speed_{}'.format(feature_type, groupby_type)] = groupby_df['{}_speed'.format(feature_type)]

    raw_feat_df['{}_a_add_xy_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_a_y_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_a_add_yz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_y_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_a_z_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_a_add_xz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_a_z_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_a_add_xyz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_a_y_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_a_z_{}'.format(feature_type, groupby_type)]

    raw_feat_df['{}_g_add_xy_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_x_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_y_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_g_add_yz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_y_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_z_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_g_add_xz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_x_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_z_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_g_add_xyz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_x_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_y_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_z_{}'.format(feature_type, groupby_type)]

    raw_feat_df['{}_ag_add_xy_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_y_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_add_yz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_y_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_z_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_add_xz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_z_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_add_zx_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_z_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_x_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_add_zy_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_z_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_y_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_add_yx_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_y_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_x_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_add_xx_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_x_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_add_yy_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_y_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_y_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_add_zz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_z_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_z_{}'.format(feature_type, groupby_type)]

    raw_feat_df['{}_a_mul_xy_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_a_y_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_a_mul_yz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_y_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_a_z_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_a_mul_xz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_a_z_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_a_mul_xyz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_a_y_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_a_z_{}'.format(feature_type, groupby_type)]

    raw_feat_df['{}_g_mul_xy_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_x_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_y_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_g_mul_yz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_y_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_z_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_g_mul_xz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_x_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_z_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_g_mul_xyz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_x_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_y_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_z_{}'.format(feature_type, groupby_type)]

    raw_feat_df['{}_ag_mul_xy_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_y_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_mul_yz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_y_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_z_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_mul_xz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_z_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_mul_zx_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_z_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_x_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_mul_zy_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_z_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_y_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_mul_yx_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_y_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_x_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_mul_xx_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_x_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_mul_yy_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_y_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_y_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_ag_mul_zz_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_z_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_z_{}'.format(feature_type, groupby_type)]
    
    return raw_feat_df

####################################################################################
########################## GROUPBY COMPRESSION FUNCTIONS ###########################
####################################################################################



####################################################################################
################################ PER SECOND METRIC #################################
####################################################################################

def get_metric_per_second_attributes(raw_feat_df, groupby_df, feature_type, groupby_type, last_second):
    
    raw_feat_df['{}_accuracy_delta_per_sec_{}'.format(feature_type, groupby_type)] = groupby_df['{}_accuracy_delta'.format(feature_type)] / list(last_second)
    raw_feat_df['{}_a_x_per_sec_{}'.format(feature_type, groupby_type)] = groupby_df['{}_a_x_delta'.format(feature_type)] / list(last_second)
    raw_feat_df['{}_a_y_per_sec_{}'.format(feature_type, groupby_type)] = groupby_df['{}_a_y_delta'.format(feature_type)] / list(last_second)
    raw_feat_df['{}_a_z_per_sec_{}'.format(feature_type, groupby_type)] = groupby_df['{}_a_z_delta'.format(feature_type)] / list(last_second)
    raw_feat_df['{}_g_x_per_sec_{}'.format(feature_type, groupby_type)] = groupby_df['{}_g_x_delta'.format(feature_type)] / list(last_second)
    raw_feat_df['{}_g_y_per_sec_{}'.format(feature_type, groupby_type)] = groupby_df['{}_g_y_delta'.format(feature_type)] / list(last_second)
    raw_feat_df['{}_g_z_per_sec_{}'.format(feature_type, groupby_type)] = groupby_df['{}_g_z_delta'.format(feature_type)] / list(last_second)
    raw_feat_df['{}_speed_per_sec_{}'.format(feature_type, groupby_type)] = groupby_df['{}_speed_delta'.format(feature_type)] / list(last_second)
    raw_feat_df['{}_bearing_per_sec_{}'.format(feature_type, groupby_type)] = groupby_df['{}_bearing_delta'.format(feature_type)] / list(last_second)

    raw_feat_df['{}_a_add_xy_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_per_sec_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_a_y_per_sec_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_a_add_yz_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_y_per_sec_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_a_z_per_sec_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_a_add_xz_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_per_sec_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_a_z_per_sec_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_a_add_xyz_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_per_sec_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_a_y_per_sec_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_a_z_per_sec_{}'.format(feature_type, groupby_type)]

    raw_feat_df['{}_g_add_xy_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_x_per_sec_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_y_per_sec_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_g_add_yz_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_y_per_sec_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_z_per_sec_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_g_add_xz_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_x_per_sec_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_z_per_sec_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_g_add_xyz_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_x_per_sec_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_y_per_sec_{}'.format(feature_type, groupby_type)] + raw_feat_df['{}_g_z_per_sec_{}'.format(feature_type, groupby_type)]

    raw_feat_df['{}_a_mul_xy_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_per_sec_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_a_y_per_sec_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_a_mul_yz_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_y_per_sec_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_a_z_per_sec_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_a_mul_xz_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_per_sec_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_a_z_per_sec_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_a_mul_xyz_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_a_x_per_sec_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_a_y_per_sec_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_a_z_per_sec_{}'.format(feature_type, groupby_type)]

    raw_feat_df['{}_g_mul_xy_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_x_per_sec_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_y_per_sec_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_g_mul_yz_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_y_per_sec_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_z_per_sec_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_g_mul_xz_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_x_per_sec_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_z_per_sec_{}'.format(feature_type, groupby_type)]
    raw_feat_df['{}_g_mul_xyz_sec_{}'.format(feature_type, groupby_type)] = raw_feat_df['{}_g_x_per_sec_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_y_per_sec_{}'.format(feature_type, groupby_type)] * raw_feat_df['{}_g_z_per_sec_{}'.format(feature_type, groupby_type)]
    
    return raw_feat_df

####################################################################################
################################ PER SECOND METRIC #################################
####################################################################################



####################################################################################
################################ STARTING FUNCTION #################################
####################################################################################

def calculate_sum_and_product_compression_interactions(compressed_data_df, groupby_df, groupby_type, last_second):
    
    ###################
    # metric attributes
    ###################
    
    compressed_data_df = get_metric_attributes(compressed_data_df, groupby_df, "q", groupby_type)
    compressed_data_df = get_metric_attributes(compressed_data_df, groupby_df, "abs", groupby_type)
    compressed_data_df = get_metric_attributes(compressed_data_df, groupby_df, "log_abs", groupby_type)
    compressed_data_df = get_metric_attributes(compressed_data_df, groupby_df, "sqr", groupby_type)
    compressed_data_df = get_metric_attributes(compressed_data_df, groupby_df, "log_sqr", groupby_type)
    
    ##############################
    # metric per second attributes
    ##############################
    
    compressed_data_df = get_metric_per_second_attributes(compressed_data_df, groupby_df, "abs", groupby_type, last_second)
    compressed_data_df = get_metric_per_second_attributes(compressed_data_df, groupby_df, "log_abs", groupby_type, last_second)
    compressed_data_df = get_metric_per_second_attributes(compressed_data_df, groupby_df, "sqr", groupby_type, last_second)
    compressed_data_df = get_metric_per_second_attributes(compressed_data_df, groupby_df, "log_sqr", groupby_type, last_second)
    
    return compressed_data_df

####################################################################################
################################ STARTING FUNCTION #################################
####################################################################################



##########################################################################
############################ MASTER FUNCTION #############################
##########################################################################

def generate_manual_features():
    
    data_df = pd.read_hdf('CLEANED_RAW_DATA/CLEANED_RAW_DATA.h5', key='CLEANED_RAW_DATA_ALL')
    
    print("Cleaned data df shape:", data_df.shape)
    
    # generate single time series features
    booking_ID_list = data_df['bookingID'].values.tolist()
    second_list = data_df['second'].values.tolist()
    p_raw = generate_TS_features(data_df.copy(), booking_ID_list, second_list)
    
    print("\nDebug: p_raw shape:", p_raw.shape)
    
    p_raw.set_index("bookingID", drop=True, inplace=True)
    
    # generate compressed features
    last_second = p_raw.groupby('bookingID').tail(1)['second']

    group_sum_df = p_raw.groupby('bookingID').sum() # time series compression by sum of changes
    group_max_df = p_raw.groupby('bookingID').max() # time series compression by max change and max change / second
    group_min_df = p_raw.groupby('bookingID').min()
    group_mean_df = p_raw.groupby('bookingID').mean()
    group_median_df = p_raw.groupby('bookingID').median()
    print("Debug: groupby shape", group_sum_df.shape, group_max_df.shape, group_min_df.shape, group_mean_df.shape, group_median_df.shape)
    
    features_df = group_sum_df
    features_df = calculate_sum_and_product_compression_interactions(features_df, group_sum_df, "GB_sum", last_second)
    features_df = calculate_sum_and_product_compression_interactions(features_df, group_max_df, "GB_max", last_second)
    features_df = calculate_sum_and_product_compression_interactions(features_df, group_min_df, "GB_min", last_second)
    features_df = calculate_sum_and_product_compression_interactions(features_df, group_mean_df, "GB_mean", last_second)
    features_df = calculate_sum_and_product_compression_interactions(features_df, group_median_df, "GB_median", last_second)
    features_df.drop(['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed'], axis=1, inplace=True)
    
    print("\nTotal generated features df shape:", features_df.shape)
    
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    print("\nNo. of NaN in data before:  ", features_df.isnull().sum().sum())
    
    features_df.dropna(axis=1, inplace=True)
    print("No. of NaN left after removal:", features_df.isnull().sum().sum())
    
    print("\nTotal cleaned generated features df shape:", features_df.shape)
    
    if not os.path.exists("GENERATED_FEATURES"): os.mkdir("GENERATED_FEATURES")
        
    features_df.to_hdf('GENERATED_FEATURES/MANUAL_FEATURES_ALL.h5', key='MANUAL_FEATURES_ALL', mode='w')

##########################################################################
############################ MASTER FUNCTION #############################
##########################################################################
    
    
    
generate_manual_features()

print("\nEnd of script 2: Manual Features Generated !")
print("End of script 2: Manual Features Generated !")
print("End of script 2: Manual Features Generated !\n")
