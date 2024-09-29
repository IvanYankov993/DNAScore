import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

class CNNMultiTaskModel:
    def __init__(self, input_shape=(250, 1), num_tasks=1, task_names=['dH', 'dS', 'dG', 'Tm']):
        self.input_shape = input_shape
        self.num_tasks = num_tasks
        self.task_names = task_names

    def r2_func_tf(self, y_true, y_pred, **kwargs):
        numerator = tf.reduce_sum(tf.square(y_true - y_pred))
        denominator = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        r2 = 1 - numerator / denominator
        return r2

    def rmse_func_tf(self, y_true, y_pred, **kwargs):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        rmse = tf.sqrt(mse)
        return rmse

    def bias_func_tf(self, y_true, y_pred, **kwargs):
        bias = tf.reduce_mean(y_true - y_pred)
        return bias

    def sdep_func_tf(self, y_true, y_pred, **kwargs):
        diff = y_true - y_pred
        mean_diff = tf.reduce_mean(diff)
        sdep = tf.sqrt(tf.reduce_mean(tf.square(diff - mean_diff)))
        return sdep

    def build_conv_layers(self, input_layer):
        """Build the convolutional layers based on hyperparameters."""
        x = input_layer

        x = layers.Conv1D(32, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv1d_1')(x)
        x = layers.MaxPooling1D(pool_size=2, name='maxpooling_1')(x)
        x = layers.BatchNormalization(name='batchnorm_1')(x)

        x = layers.Conv1D(32, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv1d_2')(x)
        x = layers.MaxPooling1D(pool_size=2, name='maxpooling_2')(x)
        x = layers.BatchNormalization(name='batchnorm_2')(x)

        # if model_type1 == "CNN3":
        #     x = layers.Conv1D(32, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv1d_3')(x)
        #     x = layers.MaxPooling1D(pool_size=2, name='maxpooling_3')(x)
        #     x = layers.BatchNormalization(name='batchnorm_3')(x)

        return x

    def build_dense_layers(self, input_layer):
        """Build the dense layers based on hyperparameters."""
        x = input_layer
        # model_type = self.hp.Choice("model_type", ["Dense3"])

        # if model_type != "Dense0":
            # hp_layer_1 = self.hp.Choice('layer_1', values=[16, 32, 64, 128])
        x = layers.Dense(64, activation='relu', kernel_initializer='glorot_uniform')(x)

        # if model_type in ["Dense2", "Dense3"]:
            # hp_layer_2 = self.hp.Choice('layer_2', values=[16, 32, 64, 128])
        x = layers.Dense(32, activation='relu', kernel_initializer='glorot_uniform')(x)

        # if model_type == "Dense3":
            # hp_layer_3 = self.hp.Choice('layer_3', values=[16, 32, 64])
        x = layers.Dense(16, activation='relu', kernel_initializer='glorot_uniform')(x)

        return x

    def build_dense_output(self, name, input_layer):
        """Build the dense output layer."""
        x = layers.Dense(1, name=name)(input_layer)
        return x

    def build_optimizer(self):
        """Build and return the optimizer with adaptive learning rate."""
        initial_learning_rate = 0.01
        decay_steps = 10.0
        decay_rate = 0.5
        learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate, decay_steps, decay_rate)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
        return optimizer

    def build_model(self):
        """Build the complete model by combining convolutional and dense layers."""
        inputs = keras.Input(shape=self.input_shape)
        x = inputs

        # Build convolutional layers
        x = self.build_conv_layers(x)

        # Flatten after convolutions
        x_flatten = layers.Flatten(name='flatten')(x)

        # Output layers for multitask learning
        outputs = []
        for i in range(self.num_tasks):
            dense_output = self.build_dense_output(
                name=self.task_names[i],
                input_layer=self.build_dense_layers(x_flatten)
            )
            outputs.append(dense_output)

        # Compile the model
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = self.build_optimizer()
        model.compile(
            loss="mse",
            optimizer=optimizer,
            metrics=["mse", 'mean_absolute_error', self.r2_func_tf, self.rmse_func_tf, self.bias_func_tf, self.sdep_func_tf]
        )

        return model
