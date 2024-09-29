from pathlib import Path
from sklearn.model_selection import ShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard

def create_train_test_split(mc_cv=50, test_frac=0.3):
    return ShuffleSplit(mc_cv, test_size=test_frac, random_state=1)

def setup_paths(home, resample, desc, model_name):
    # Define the directory path
    directory_path = Path(f"{home}/CV/{resample}/{desc}/{model_name}/")

    # Define Paths for call backs
    csv_logger_path  = directory_path / 'csv_logger/'
    cp_callback_path = directory_path / 'model_checkpoint/'
    tensorboard_path = directory_path / 'tensorboard_logs/'

    # Ensure the directory exists, create it if necessary
    csv_logger_path.mkdir(parents=True, exist_ok=True)
    cp_callback_path.mkdir(parents=True, exist_ok=True)
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    return csv_logger_path, cp_callback_path, tensorboard_path

def setup_callbacks(csv_logger_path, cp_callback_path, tensorboard_path):
    es = EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        verbose=1, 
        patience=2000, 
        restore_best_weights=True
    )
    
    csv_logger = CSVLogger(f'{csv_logger_path}/model_history.csv', append=True)

    # Uncomment if using model checkpoints
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f'{cp_callback_path}/cp.ckpt',
    #     save_weights_only=True,
    #     verbose=1
    # )

    tensorboard_callback = TensorBoard(
        log_dir=tensorboard_path, 
        update_freq=1,
        write_graph=False, 
        write_images=False
    )

    # Returning the callbacks as a list
    return [es, csv_logger, tensorboard_callback]

def train_model(model, X_train, y_train, X_val, y_val, prop, home, resample, desc, model_name, epochs=200, batch_size=32):
    # Setup paths
    csv_logger_path, cp_callback_path, tensorboard_path = setup_paths(home, resample, desc, model_name)
    
    # Setup callbacks
    keras_callbacks = setup_callbacks(csv_logger_path, cp_callback_path, tensorboard_path)
    
    # Training the model
    history = model.fit(
        X_train, 
        y_train, 
        validation_data=(X_val, y_val), 
        epochs=epochs, 
        batch_size=batch_size, 
        verbose=2,
        callbacks=keras_callbacks
    )
    
    return history


#
#
#
#
#
#
#
#
#
#
#
#
#


import os
from sklearn.model_selection import ShuffleSplit
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Function to create callbacks
def create_callbacks(save_dir, model_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    checkpoint_path = os.path.join(save_dir, f'{model_name}_best.h5')
    log_dir = os.path.join(save_dir, "logs")

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    return [checkpoint, early_stopping, reduce_lr, tensorboard]

# Function to train the model
def train_model(model, X_train, y_train, X_val, y_val, save_dir, model_name="model", epochs=200, batch_size=32):
    # Create callbacks
    callbacks = create_callbacks(save_dir, model_name)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=2
    )
    
    return history

# Function to create train-test split
def create_train_test_split(mc_cv=50, test_frac=0.3):
    return ShuffleSplit(mc_cv, test_size=test_frac, random_state=1)
