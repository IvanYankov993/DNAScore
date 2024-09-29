import numpy as np
import pandas as pd
import pathlib
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('agg')
import glob
import random

# import tensorflow as tf
# import keras_tuner as kt
# from tensorflow import keras
# from tensorflow.keras import Input, Model
# from tensorflow.keras import layers, models, initializers, optimizers
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

# # Check if these are in model
# from tensorflow.keras.losses import Reduction 
# # from tensorflow import tensorflow.keras.losses.Reduction
# # from tensorflow.keras.losses import Loss
# from tensorflow.keras.losses import MeanAbsoluteError
# # Finish check


# from scikeras.wrappers import KerasRegressor

# import sklearn
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn import metrics
from sklearn import datasets
from sklearn.metrics import make_scorer


import os
import pathlib


from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle as pk

def padding(X_descr_train_scaled):
#     Padding function so X data is always 250 dimensions
# Must be coupled with load_data. NB! double check if the scalling is not affected
# https://www.geeksforgeeks.org/python-call-function-from-another-function/
    a=X_descr_train_scaled.to_numpy()
    b=np.zeros((len(X_descr_train_scaled), 
                (250-int(X_descr_train_scaled.to_numpy().shape[1]))
               )
              )
    padded=np.concatenate((a,b),
                           axis=1, 
                          out=None, 
                          dtype=None
                         )
    return padded


def df_np(y):
#     y is a list
    y_out=[]
    for y_i in y:
        y_ic=y_i.to_numpy()
        y_ic=y_ic.reshape(y_ic.shape[0])
        y_out.append(y_ic)
    return y_out

def load_data(file,prop):
# Universal funciton for loading
# y_1, y_2, y_3, y_4 and x data from input csv (All, Train, Val or Train)
    y_1 = file[['dH']].copy()
    y_2 = file[['dS']].copy()
    y_3 = file[['dG']].copy()
    y_4 = file[['Tm']].copy()
    
    # Convert y data into required input shape
    y_1 = y_1.to_numpy()
    y_1 = y_1.reshape(y_1.shape[0])
    y_2 = y_2.to_numpy()
    y_2 = y_2.reshape(y_2.shape[0])
    y_3 = y_3.to_numpy()
    y_3 = y_3.reshape(y_3.shape[0])
    y_4 = y_4.to_numpy()
    y_4 = y_4.reshape(y_4.shape[0])
    
    # Load features based on prop
    X = file[[col for col in file.columns if f'{prop}_'in col]]
    
    return y_1, y_2, y_3, y_4, padding(X), X

def load_data_df(file,prop):
# Universal funciton for loading
# y_1, y_2, y_3, y_4 and x data from input csv (All, Train, Val or Train)
    y_1 = file[['dH']].copy()
    y_2 = file[['dS']].copy()
    y_3 = file[['dG']].copy()
    y_4 = file[['Tm']].copy()
    
    # Load features based on prop
    X = file[[col for col in file.columns if f'{prop}_'in col]]
    
    return y_1, y_2, y_3, y_4, X


def wrapped_train_test_split(train_idx,test_idx,file,prop):
#     capittal Y and X stand for pandas dataframe like file
    Y_1, Y_2, Y_3, Y_4, X = load_data_df(file,prop)

    # Separate data into training and test sets:
    x_train = X.iloc[train_idx]
    x_test = X.iloc[test_idx]
#     The next two lines (y) will vary depending on the CNN output
    y_train = [Y_1.iloc[train_idx],Y_2.iloc[train_idx],Y_3.iloc[train_idx],Y_4.iloc[train_idx]]
    y_test  = [Y_1.iloc[test_idx] ,Y_2.iloc[test_idx] ,Y_3.iloc[test_idx] ,Y_4.iloc[test_idx]]
    
    return padding(x_train), padding(x_test), df_np(y_train), df_np(y_test)

def wrapped_train_test_split_no_padding(train_idx,test_idx,file,prop):
#     capittal Y and X stand for pandas dataframe like file
    Y_1, Y_2, Y_3, Y_4, X = load_data_df(file,prop)

    # Separate data into training and test sets:
    x_train = X.iloc[train_idx]
    x_test = X.iloc[test_idx]
#     The next two lines (y) will vary depending on the CNN output
    y_train = [Y_1.iloc[train_idx],Y_2.iloc[train_idx],Y_3.iloc[train_idx],Y_4.iloc[train_idx]]
    y_test  = [Y_1.iloc[test_idx] ,Y_2.iloc[test_idx] ,Y_3.iloc[test_idx] ,Y_4.iloc[test_idx]]
    
    return x_train, x_test, df_np(y_train), df_np(y_test)

def r2_func(y_true, y_pred, **kwargs):
    return metrics.r2_score(y_true, y_pred)
def rmse_func(y_true, y_pred, **kwargs):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))  
def bias_func(y_true, y_pred, **kwargs):
    return np.mean(y_true-y_pred)
def sdep_func(y_true, y_pred, **kwargs):
    return (np.mean((y_true-y_pred-(np.mean(y_true-y_pred)))**2))**0.5
#these 4 are for tensorflow formats
def r2_func_tf(y_true, y_pred, **kwargs):
    numerator = tf.reduce_sum(tf.square(y_true - y_pred))
    denominator = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - numerator / denominator
    return r2
def rmse_func_tf(y_true, y_pred, **kwargs):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    rmse = tf.sqrt(mse)
    return rmse
def bias_func_tf(y_true, y_pred, **kwargs):
    bias = tf.reduce_mean(y_true - y_pred)
    return bias
def sdep_func_tf(y_true, y_pred, **kwargs):
    diff = y_true - y_pred
    mean_diff = tf.reduce_mean(diff)
    sdep = tf.sqrt(tf.reduce_mean(tf.square(diff - mean_diff)))
    return sdep

def save_splits(idx,true,pred,resample,path,split_type):
    y_outputs = pd.DataFrame()
    y_outputs['ID'] = idx
    y_outputs['y_true'] = true
    y_outputs['y_pred'] = pred
    y_outputs.to_csv(f'{path}/Split_{resample}_type_{split_type}.csv', index=False)
    return

def create_dir(home,resample,model_name,prop,GSHT):
    if home==None:
        home=os.getcwd()
    try:
            os.mkdir("{}/CV/".format(home))
    except:
            pass
    os.chdir("{}/CV/".format(home))
    try:
        os.mkdir("{}".format(resample))
    except:
            pass
    os.chdir("{}".format(resample))
    try:
        os.mkdir("{}".format(model_name))
    except:
            pass
    os.chdir("{}".format(model_name))
    try:
        os.mkdir("{}".format(prop))
    except:
            pass
    os.chdir("{}".format(prop))
    try:
        os.mkdir("{}".format(GSHT))
    except:
            pass
    os.chdir("{}".format(GSHT))

    os.chdir("{}".format(home))
    return

__file__=os.path.abspath('CNN_jony')
rootdir = pathlib.Path(__file__).parent
datadir = rootdir/"Input"
resultdir = rootdir/"Outputs"


# Initialise
df_all=pd.DataFrame()
df1=pd.read_csv('resample_1_test_resample.csv')
df2=pd.read_csv('resample_1_train_resample.csv')
df3=pd.read_csv('resample_1_val_resample.csv')
# concatenate
df_all=pd.concat([df1,df2,df3])
# Check shape
print(df_all.shape, df1.shape, df2.shape, df3.shape)
# organise index
df_all=df_all.set_index('Index').sort_index()
# Save all data
df_all.to_csv("Lomzov_dataset_IY.csv",index=False)
# Load data from a CSV file
test_df=pd.read_csv("Lomzov_dataset_IY.csv")
# parameters to work with
file=test_df
prop='Granulated'
# obtain y and x data
y_1, y_2, y_3, y_4, x, X= load_data(file,prop)


# input_dataset = pd.read_csv(f'{root_dir}/Descriptor_sets/Descriptors_Clusters_Y.csv')

desc_type = ['RF-Score','H-Bonding','Granulated','DNA-Groups','OHEP','LP_dec2','CountDNA','CountDNAp']

no_resamples = 1

cv_dir=["random_CV","stratified_CV"]

models_list = ["RF","KNN","st_1DCNN","mt3_1DCNN","mt4_1DCNN"]

resample=1

fold=2

# resample_folder=(f'{root_dir}/random_CV/resample_{resample}/resample_{resample}')

# fold_folder=(f'{root_dir}/random_CV/resample_{resample}/fold_{fold}')



###set global variables
#number of resamples, sfed_type, n_jobs, folds (might be redundant), epochs
n_resamples = 1
n_jobs = 1
# same
n_folds = 2
mc_cv = 2
# train test split
test_frac = 0.3
# 
epochs = 20
