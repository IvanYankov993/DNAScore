import tensorflow as tf
import numpy as np
import sys

# Define the custom metric function
def r2_func_tf(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())

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

# Load the model with the custom metric
custom_objects = {'r2_func_tf': r2_func_tf,
                  'rmse_func_tf': rmse_func_tf,
                  'bias_func_tf': bias_func_tf,
                  'sdep_func_tf': sdep_func_tf,}

# Disable eager execution (optional, but can help in non-interactive environments)
tf.compat.v1.disable_eager_execution()


# Load the trained model (SavedModel format or .h5 file)
model = tf.keras.models.load_model('DNA_score_savedmode',custom_objects=custom_objects)
@tf.autograph.experimental.do_not_convert
def predict_input(model, X_new):
    return model.predict(X_new, verbose=0)

# Get the input data from the command line (passed as a string)
input_str = sys.argv[1]

# Convert the input back to a NumPy array (adjust based on your input format)
X_new = np.fromstring(input_str, sep=',')
X_new = X_new.reshape(1, -1, 1)  # Reshape as necessary

# Make predictions
predictions = predict_input(model, X_new)

# Print the predictions so the calling process can capture the output
print(predictions.tolist(), flush=True)

