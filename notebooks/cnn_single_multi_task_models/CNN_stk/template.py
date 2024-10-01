# Part one General and Data Hanlding

import numpy as np
import pandas as pd
import os
from pathlib import Path

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

# Part 2 Model

import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras import layers, models, initializers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger



# Data Block Functions

def path_fold(home,resample,i_fold):
    path="{}/CV/{}/fold_{}".format(os.getcwd(),resample,i_fold)
        
    # Define the directory path
    directory_path = Path(f"{home}/CV/{resample}/{i_fold}")
    
    # Ensure the directory exists, create it if necessary
    directory_path.mkdir(parents=True, exist_ok=True)

    return directory_path

def path_resample(home,resample):
    path="{}/CV/{}/".format(os.getcwd(),resample)
        
    # Define the directory path
    directory_path = Path(f"{home}/CV/{resample}")
    
    # Ensure the directory exists, create it if necessary
    directory_path.mkdir(parents=True, exist_ok=True)

    return directory_path

def cv_hp(df,home):
    resample_split  = ShuffleSplit(50, test_size=0.3, random_state=1)
    fold_split      = ShuffleSplit(5 , test_size=0.3, random_state=1)
    train_val_split = ShuffleSplit(1 , test_size=0.3, random_state=1)
    
    for resample, (train_val_index, test_index) in enumerate(resample_split.split(df)):
        train_val = pd.DataFrame(df['ID'].iloc[train_val_index])
        test = pd.DataFrame(df['ID'].iloc[test_index])
        for i, (train_index, val_index) in enumerate(train_val_split.split(train_val)):
            train = pd.DataFrame(df['ID'].iloc[train_index])
            val   = pd.DataFrame(df['ID'].iloc[val_index])
        resample_path = path_resample(home,resample)
        train.to_csv(f'{resample_path}/train.csv')
        val.to_csv(f'{resample_path}/val.csv')
        test.to_csv(f'{resample_path}/test.csv')
        # train,val,test to_csv
        for i_fold, (train_val_fold_index, test_fold_index) in enumerate(fold_split.split(train)):
            train_val_fold = pd.DataFrame(train['ID'].iloc[train_val_fold_index])
            test_fold = pd.DataFrame(train['ID'].iloc[test_fold_index])
            for i, (train_fold_index, val_fold_index) in enumerate(train_val_split.split(train_val_fold)):
                train_fold = pd.DataFrame(train_val_fold['ID'].iloc[train_fold_index])
                val_fold   = pd.DataFrame(train_val_fold['ID'].iloc[val_fold_index])
            i_fold_path = path_fold(home,resample,i_fold)
            train_fold.to_csv(f'{i_fold_path}/train.csv')
            val_fold.to_csv(f'{i_fold_path}/val.csv')
            test_fold.to_csv(f'{i_fold_path}/test.csv')
            

    return print("data organised into 50 CV with 5-fold inner CV")


# Accessing Data Via ID CV and Full table functions

def access_fold_csv(df,home,resample,fold):
    df_path = path_fold(home,resample,fold)
    train_df=pd.read_csv(f'{df_path}/train.csv')
    val_df=pd.read_csv(f'{df_path}/val.csv')
    test_df=pd.read_csv(f'{df_path}/test.csv')

    train_df=df[df["ID"].isin(train_df['ID'])]
    val_df=df[df["ID"].isin(val_df['ID'])]
    test_df=df[df["ID"].isin(test_df['ID'])]
    return train_df, val_df, test_df


def access_resample_csv(df,home,resample):
    df_path = path_resample(home,resample)
    train_df=pd.read_csv(f'{df_path}/train.csv')
    val_df=pd.read_csv(f'{df_path}/val.csv')
    test_df=pd.read_csv(f'{df_path}/test.csv')

    train_df=df[df["ID"].isin(train_df['ID'])]
    val_df=df[df["ID"].isin(val_df['ID'])]
    test_df=df[df["ID"].isin(test_df['ID'])]
    return train_df, val_df, test_df


# Loading X and Y Data Functions

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


def load_xy(file,desc):
    # Universal funciton for loading
# y_1, y_2, y_3, y_4 and x data from input csv (All, Train, Val or Train)
    y_1 = file[['dH']].copy()
    y_2 = file[['dS']].copy()
    y_3 = file[['dG']].copy()
    y_4 = file[['Tm']].copy()

    Y = file[['dH','dS','dG','Tm']].copy()
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
    X = file[[col for col in file.columns if f'{desc}_'in col]]
    
    return y_1, y_2, y_3, y_4, Y, padding(X), X



# Model and Scoring Functions

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

# Model


def build_model(hp):
    
#     Hyper parameters         
    model_type1 = hp.Choice("model_type1", ["CNN3","CNN2","CNN1"])
    model_type = hp.Choice("model_type", ["Dense3"])
 
    
#     INPUT for NN
    
    inputs = keras.Input(shape=(250,1))
    x_layer=inputs
    
#     MANDATORY CNN (optional to move into first condition hp.cond_scope
    # with hp.conditional_scope("model_type1", ["CNN0"]):
    #         if model_type1 == "CNN0":
    #             pass
#     CONDITIONAL CONVOLUTION LAYERS (Consider moving the above into CNN1) test 0-3 CNN and 0-3 Dense
    with hp.conditional_scope("model_type1", ["CNN1","CNN2""CNN3"]):
            if model_type1 != "CNN0":
                x_layer = keras.layers.Conv1D(32, 
                        kernel_size=(3), 
                        strides=(2), 
                        padding='valid', 
                        activation='relu', 
                        input_shape=(250,1),
                        name = 'conv1d_1'
                        )(x_layer)
                x_layer = keras.layers.MaxPooling1D((2), name = 'maxpooling_1')(x_layer)
                x_layer = keras.layers.BatchNormalization(name = 'batchnorm_1')(x_layer)
                pass
                
            if model_type1 != "CNN1":
                x_layer = keras.layers.Conv1D(32, 
                                    kernel_size=(3), 
                                    strides=(2), 
                                    padding='valid', 
                                    activation='relu', 
                                    name = f'conv1d_2'
                                    )(x_layer)
                x_layer = keras.layers.MaxPooling1D((2), name = f'maxpooling_2')(x_layer)
                x_layer = keras.layers.BatchNormalization(name = f'batchnorm_2')(x_layer)

            if model_type1 != "CNN1" or "CNN2":               
                x_layer = keras.layers.Conv1D(32, 
                                    kernel_size=(3), 
                                    strides=(2), 
                                    padding='valid', 
                                    activation='relu', 
                                    name = f'conv1d_3'
                                    )(x_layer)
                x_layer = keras.layers.MaxPooling1D((2), name = f'maxpooling_3')(x_layer)
                x_layer = keras.layers.BatchNormalization(name = f'batchnorm_3')(x_layer)
                
#     FLATTEN AFTER CONVOLUTIONS
    x_layer = keras.layers.Flatten(name = 'flatten')(x_layer)
    
#     CONDITIONAL DENSE LAYERS
    # with hp.conditional_scope("model_type", ["Dense0"]):
    #     if model_type == "Dense0":
    #         pass
            
    with hp.conditional_scope("model_type", ["Dense3"]): #["Dense1","Dense2","Dense3"]
        if model_type != "Dense0":
            hp_layer_1= hp.Choice(f'layer_1', values=[16,32,64,128])

            x_layer = keras.layers.Dense(
                        hp_layer_1,
                        activation='relu',
                        use_bias=True,
                        # name='layer_1',
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None
                    )(x_layer)
        if model_type != "Dense1":
            hp_layer_2_2= hp.Choice(f'layer_2_2', values=[16,32,64,128])

            x_layer = keras.layers.Dense(
                        hp_layer_2_2,
                        activation='relu',
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None
                    )(x_layer)

        if model_type != "Dense1" or "Dense2":
            hp_layer_3_3= hp.Choice(f'layer_3_3',  values=[16,32,64])
            
            x_layer = keras.layers.Dense(
                        hp_layer_3_3,
                        activation='relu',
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None
                    )(x_layer)
#     OUTPUT LAYERS

    # output_1 = keras.layers.Dense(1, name='enthalpy_pred')(x_layer)
    # output_2 = keras.layers.Dense(1, name='entropy_pred')(x_layer)
    # output_3 = keras.layers.Dense(1, name='free_energy_pred')(x_layer)
    # output_4 = keras.layers.Dense(1, name='melting_temperature')(x_layer)

    # output_1 = keras.layers.Dense(1, name='dH')(x_layer)
    # output_2 = keras.layers.Dense(1, name='dS')(x_layer)
    # output_3 = keras.layers.Dense(1, name='dG')(x_layer)
    # output_4 = keras.layers.Dense(1, name='Tm')(x_layer)
    

    # model = Model(inputs=inputs, outputs=[output_1, output_2, output_3, output_4])
    output_1 = keras.layers.Dense(1, name='output')(x_layer)
    model = Model(inputs=inputs, outputs=output_1)
    
#     SETTINGS
#     SETTINGS

#     ADAPTIVE LEARNING RATE   
    
    initial_learning_rate = 0.01
    decay_steps = 10.0
    decay_rate = 0.5
    learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(
                                    initial_learning_rate, decay_steps, decay_rate)
    
#     SETTING ADAM OPTIMISER
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    
#     COMPILE MODEl
    model.compile(loss = "mse" , 
                  optimizer = optimiser, 
                  metrics = ["mse",'mean_absolute_error',r2_func_tf, rmse_func_tf, bias_func_tf, sdep_func_tf])   
    
    return model


# Code Execution 

df=pd.read_csv("Lomzov_dataset_IY.csv")
home=os.getcwd()
# data Generation
# cv_hp(df,home)

# Actual instrucitons
resample=1
fold=1
desc='RF-Score'
prop = 'dH' 
model_name = f"1DConv_st_{prop}" 
epochs = 300
batch  = 16

# train, val, test = access_resample_csv(df,home,resample)
train_fold, val_fold, test_fold = access_fold_csv(df,home,resample,fold)

y_1_train, y_2_train, y_3_train, y_4_train, Y_train, X_padded_train, X_train = load_xy(train_fold,desc)
y_1_val,   y_2_val,   y_3_val,   y_4_val,   Y_val,   X_padded_val,   X_val   = load_xy(val_fold,desc)
# y_1_test,  y_2_test,  y_3_test,  y_4_test,  Y_test,  X_padded_test,  X_test  = load_xy(test_fold,desc)

fold_path = path_fold(home,resample,fold)
# resample_path = path_resample(home,resample)

# Define the directory path
directory_path = Path(f"{fold_path}/{desc}/{model_name}/")

tunner_path      = Path(f'{directory_path}/tunner')
csv_logger_path  = Path(f'{directory_path}/csv_logger/')
cp_callback_path = Path(f'{directory_path}/model_checkpoint/')
tensorboard_path = Path(f'{directory_path}/tensorboard_logs/')

# Ensure the directory exists, create it if necessary
tunner_path.mkdir(parents=True, exist_ok=True)
csv_logger_path.mkdir(parents=True, exist_ok=True)
cp_callback_path.mkdir(parents=True, exist_ok=True)
tensorboard_path.mkdir(parents=True, exist_ok=True)


tuner = kt.GridSearch(build_model,
                   objective=kt.Objective('val_loss', 'min'),
                    # loss = 'val_loss',
                   # objective = ['val_mse','val_free_energy_pred_mse'],
                  directory=tunner_path,
                  overwrite=False,
                  project_name=f'{batch}')

with open(f'{tunner_path}/tuner_path.txt', 'w') as f:
    f.write(tuner.project_dir)
f.close

#### CALL BACKS!
es = EarlyStopping(monitor      = 'val_loss', 
                        mode     = 'min', 
                        verbose  = 1, 
                        patience = 2000, 
                    restore_best_weights = True)
# CSV Logger
csv_logger = CSVLogger(f'{csv_logger_path}/model_history.csv' , append=True)

# CP_callbacks      not required when using a tunner       
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'{cp_callback_path}/cp.ckpt',
#                                                  save_weights_only=True,
#                                                  verbose=1)

# TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, 
                                                       #/{batch}', # _ADAPTIVELEARNIGNRATE_01_10_Dense3_64_3CNN_lr_3_es
                                                      update_freq = 1,
                                                      # histogram_freq=1, 
                                                      write_graph=False, 
                                                      write_images=False)
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard

# Covert to list and provide to Keras Regressor
keras_callbacks = [es, csv_logger, tensorboard_callback]

# Execution

history=tuner.search(X_padded_train, y_1_train,
            epochs = epochs,
            batch_size=batch,
            verbose = 3,
            validation_data =(X_padded_val, y_1_val),
             # validation_split = 0.2,
            callbacks=keras_callbacks)
