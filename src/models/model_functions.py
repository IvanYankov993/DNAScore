
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model




# def r2_func(y_true, y_pred, **kwargs):
#     return metrics.r2_score(y_true, y_pred)
# def rmse_func(y_true, y_pred, **kwargs):
#     return np.sqrt(metrics.mean_squared_error(y_true, y_pred))  
# def bias_func(y_true, y_pred, **kwargs):
#     return np.mean(y_true-y_pred)
# def sdep_func(y_true, y_pred, **kwargs):
#     return (np.mean((y_true-y_pred-(np.mean(y_true-y_pred)))**2))**0.5

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

def build_conv_layers(hp, input_layer):
    """Build the convolutional layers based on hyperparameters."""
    x = input_layer
    model_type1 = hp.Choice("model_type1", ["CNN3","CNN2","CNN1"])

    if model_type1 != "CNN0":
        x = layers.Conv1D(32, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv1d_1')(x)
        x = layers.MaxPooling1D(pool_size=2, name='maxpooling_1')(x)
        x = layers.BatchNormalization(name='batchnorm_1')(x)

    if model_type1 in ["CNN2", "CNN3"]:
        x = layers.Conv1D(32, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv1d_2')(x)
        x = layers.MaxPooling1D(pool_size=2, name='maxpooling_2')(x)
        x = layers.BatchNormalization(name='batchnorm_2')(x)

    if model_type1 == "CNN3":
        x = layers.Conv1D(32, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv1d_3')(x)
        x = layers.MaxPooling1D(pool_size=2, name='maxpooling_3')(x)
        x = layers.BatchNormalization(name='batchnorm_3')(x)

    return x

def build_dense_layers(hp, input_layer):
    """Build the dense layers based on hyperparameters."""
    x = input_layer
    model_type = hp.Choice("model_type", ["Dense3"])

    if model_type != "Dense0":
        hp_layer_1 = hp.Choice('layer_1', values=[16,32,64,128])
        x = layers.Dense(hp_layer_1, activation='relu', kernel_initializer='glorot_uniform')(x)

    if model_type in ["Dense2", "Dense3"]:
        hp_layer_2 = hp.Choice('layer_2', values=[16,32,64,128])
        x = layers.Dense(hp_layer_2, activation='relu', kernel_initializer='glorot_uniform')(x)

    if model_type == "Dense3":
        hp_layer_3 = hp.Choice('layer_3', values=[16,32,64])
        x = layers.Dense(hp_layer_3, activation='relu', kernel_initializer='glorot_uniform')(x)

    return x


def build_dense_output(name='output', input_layer):
    """Build the dense layers based on hyperparameters."""
    x = input_layer
    
    x = keras.layers.Dense(1, name=name)(x)

    return x

def build_optimizer():
    """Build and return the optimizer with adaptive learning rate."""
    initial_learning_rate = 0.01
    decay_steps = 10.0
    decay_rate = 0.5
    learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate, decay_steps, decay_rate)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    return optimizer

def build_model(hp,num_tasks=1,names=['dH','dS','dG','Tm']):
    """Build the complete model by combining convolutional and dense layers."""
    inputs = keras.Input(shape=(250,1))
    x = inputs

    # Build convolutional layers
    x = build_conv_layers(hp, x)

    # Flatten after convolutions
    x_flatten = layers.Flatten(name='flatten')(x)

    # Build dense layers
    # x = build_dense_layers(hp, x_flatten)

    # Output layer
    outputs = []
    
    for i in range(1, num_tasks + 1):
        output = build_dense_output(name=names[i-1],
                                    input_layer=build_dense_layers(hp, x_flatten))
        outputs.append(output)

    # Compile the model
    model = Model(inputs=inputs, outputs=output)
    optimizer = build_optimizer()
    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics = ["mse",'mean_absolute_error',r2_func_tf, rmse_func_tf, bias_func_tf, sdep_func_tf])   
        # Add other custom metrics like r2_func_tf, rmse_func_tf, etc.
    
    return model


# Explanation of Changes:
# Modular Functions:
# build_conv_layers: Handles the creation of convolutional layers.
# build_dense_layers: Handles the creation of dense layers.
# build_optimizer: Manages the optimizer and learning rate schedule.
# Cleaner build_model Function:
# The build_model function now simply orchestrates the overall model-building process, using the modular components.
# Improved Conditional Logic:
# Conditions are streamlined for clarity and avoid redundant checks (e.g., the previous conditional was checked multiple times).
# Using This Structure:
# When building models in your training scripts or notebooks, you can now easily call build_model(hp) and have a well-structured, maintainable, and modular model-building process.

# Would you like to proceed with further refinements or testing this refactored structure in your training setup?