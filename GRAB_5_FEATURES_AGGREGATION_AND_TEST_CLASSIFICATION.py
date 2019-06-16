#!/home/ubuntu/anaconda3/bin/python3.7

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from lightgbm import LGBMClassifier
import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')

print("\nStarting script 5: Features aggregation, features filtering and test prediction")
print("Starting script 5: Features aggregation, features filtering and test prediction")
print("Starting script 5: Features aggregation, features filtering and test prediction\n")

print("Warning: Step 4 tsFresh module requires pandas 0.23.4 version. If encounter error use pip install pandas to automatically upgrade pandas back to 0.24.2")
print("Warning: Step 4 tsFresh module requires pandas 0.23.4 version. If encounter error use pip install pandas to automatically upgrade pandas back to 0.24.2")
print("Warning: Step 4 tsFresh module requires pandas 0.23.4 version. If encounter error use pip install pandas to automatically upgrade pandas back to 0.24.2\n")

##########################################################################
###################### Loading selected features mask ####################
##########################################################################

features_filter_manual = pd.read_csv("UTILITY/MANUAL_SELECTED_FEATURES.csv")
features_filter_lstm = pd.read_csv("UTILITY/LSTM_SELECTED_FEATURES.csv")
features_filter_tsfresh = pd.read_csv("UTILITY/TSFRESH_SELECTED_FEATURES.csv")
features_filter_concated = pd.concat([features_filter_manual['Feature'], features_filter_lstm['Feature'], features_filter_tsfresh['Feature']], axis=0, ignore_index=True)
features_filter_concated_array = features_filter_concated.values

print("Feature mask shapes:", features_filter_manual.shape, features_filter_lstm.shape, features_filter_tsfresh.shape, features_filter_concated.shape)

##########################################################################
###################### Loading selected features mask ####################
##########################################################################



##########################################################################
####### Load extracted features & filter in selected features only #######
##########################################################################

print("\nLoading generated features...")

extracted_features_manual_df = pd.read_hdf('GENERATED_FEATURES/MANUAL_FEATURES_ALL.h5').sort_index()
extracted_features_lstm_df = pd.read_hdf('GENERATED_FEATURES/LSTM_FEATURES_ALL.h5').sort_index()
extracted_features_tsfresh_df = pd.read_hdf('GENERATED_FEATURES/TSFRESH_FEATURES_ALL.h5').sort_index()
extracted_features_concated = pd.concat([extracted_features_manual_df, extracted_features_lstm_df, extracted_features_tsfresh_df], axis=1)

print("\nExtracted features shapes:", extracted_features_manual_df.shape, extracted_features_lstm_df.shape, extracted_features_tsfresh_df.shape, extracted_features_concated.shape)
# print(extracted_features_lstm_df.index)
# print(extracted_features_tsfresh_df.index)
# print(extracted_features_manual_df.index)

features_all_df = extracted_features_concated.loc[:, features_filter_concated_array]
print("\nSelected extracted features shape:", features_all_df.shape)

##########################################################################
####### Load extracted features & filter in selected features only #######
##########################################################################



##########################################################################
####################  Feature - Label Train Test Split ###################
##########################################################################

labels_all_df = pd.read_hdf('CLEANED_RAW_DATA/CLEANED_RAW_DATA.h5', key='CLEANED_LABELS_ALL').sort_index()

print("")
print("Debug:", features_all_df.shape, labels_all_df.shape)

features_all_df['label'] = labels_all_df['label']

print("Debug:", features_all_df.shape, labels_all_df.shape)

features_all_df = features_all_df.replace([np.inf, -np.inf], np.nan)

print("Debug:", features_all_df.shape, labels_all_df.shape)

test_both_df = features_all_df
X_test = test_both_df.iloc[:, :-1]
y_test = test_both_df.iloc[:, -1:]

print("")
print("Test Input Features Shape: {}     Test Label Shape: {}".format(X_test.shape, y_test.shape))
print("")

##########################################################################
####################  Feature - Label Train Test Split ###################
##########################################################################



##########################################################################
################# Loading classifiers and test prediction ################
##########################################################################

cpp_lgbm_classifier_list = pickle.load(open("UTILITY/clf_cpp_lgbm_protocol_3.pickle", "rb"))
sklearn_lgbm_classifier_list = pickle.load(open("UTILITY/clf_sklearn_lgbm_protocol_3.pickle", "rb"))

prediction_list = []

for index, clf in enumerate(cpp_lgbm_classifier_list):
    print("Predicting with {} / {} c++ lightGBM model".format(index+1, len(cpp_lgbm_classifier_list)))
    prediction_list.append(clf.predict(X_test))

for index, clf in enumerate(sklearn_lgbm_classifier_list):
    print("Predicting with {} / {} sklearn lightGBM model".format(index+1, len(sklearn_lgbm_classifier_list)))
    prediction_list.append(clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:, 1])

average_prediction_probability = np.mean(np.array(prediction_list), axis=0)

pd.Series(average_prediction_probability).to_csv("Test_Prediction_Probabilities.csv")
print("\nPredictions probability saved in file: Test_Prediction_Probabilities.csv")

auc_score = roc_auc_score(y_test, average_prediction_probability)

print("\nTest AUC score:", auc_score)

false_positive_rate, true_positive_rate, _ = roc_curve(y_test, average_prediction_probability)

plt.figure()
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(false_positive_rate, true_positive_rate, marker='.', label='AUC: {:.3f}'.format(auc_score))
plt.title('ROC AUC Plot')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("ROC_AUC_Plot.jpg", dpi=128)

print("\nROC AUC plot exported to: ROC_AUC_Plot.jpg")

print("\nEnd of script 5: Test Predictions Done !")
print("End of script 5: Test Predictions Done !")
print("End of script 5: Test Predictions Done !\n")

##########################################################################
################# Loading classifiers and test prediction ################
##########################################################################
