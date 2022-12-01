import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import layers


# _, n_timesteps, n_channels = X_train_nn.shape
# output_units = y_train_nn.shape[-1]
# print(f"input_shape = {(n_timesteps, n_channels)} | output_units = {output_units}")

# input_channels = x = tf.keras.layers.Input(shape=(n_timesteps, n_channels))

# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.SimpleRNN(
#     units=100,
#     return_sequences=True,
# )(x)
# x = tf.keras.layers.BatchNormalization()(x)

# output = tf.keras.layers.Dense(units=output_units, activation='sigmoid')(x)

# model = tf.keras.Model(
#     inputs=input_channels,
#     outputs=output,
#     name="Model"
# )

# # Отображение конфигурации модели
# model.summary()# Отображение конфигурации модели в виде таблицы

class simpleRNN():
    def __init__(self, X_train_nn, y_train_nn):
        self.n_timesteps = X_train_nn.shape[1]
        self.n_channels = X_train_nn.shape[2]
        self.output_units = y_train_nn.shape[-1]
        print(f"input_shape = {(self.n_timesteps, self.n_channels)} | output_units = {self.output_units}")


    def build_model (self, units):
        input_channels = x = tf.keras.layers.Input(shape=(self.n_timesteps, self.n_channels))

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.SimpleRNN(
            units=units,
            return_sequences=True,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(units=self.output_units, activation='sigmoid')(x)

        model = tf.keras.Model(
            inputs=input_channels,
            outputs=output,
            name="Model"
        ) 
        print(model.summary)