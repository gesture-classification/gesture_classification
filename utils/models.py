import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import layers

# Импорт параметров
from utils.functions import config_reader, f1

config = config_reader() #'../config/data_config.json'

class SimpleRNN():
    """Класс создаёт модель simpleRNN.
    Параметры:
    ----------
    n_timesteps (_int_) - кол-во временных периодов
    n_channels (_int_) - кол-во датчиков
    output_units - 
    units - размерность модели   
    """    
    def __init__(self, X_train_nn, y_train_nn, units=config['simpleRNN_units']):
        super(SimpleRNN, self).__init__()
        self.n_timesteps = None #X_train_nn.shape[1]
        self.n_channels = X_train_nn.shape[2]
        self.output_units = y_train_nn.shape[-1]
        self.units = units
        self.loss = "mean_squared_error"
        print(f"input_shape = {(self.n_timesteps, self.n_channels)} | output_units = {self.output_units}")
               

    def build_model(self):
        """Метод формирования модели
        """
        input_channels = x = tf.keras.layers.Input(shape=(self.n_timesteps, self.n_channels))

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.SimpleRNN(
            units=self.units,
            return_sequences=True,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(units=self.output_units, activation='sigmoid')(x)

        model = tf.keras.Model(
            inputs=input_channels,
            outputs=output,
            name="model"
        ) 
        print(model.summary())
        return model
    
    def compile(self, model):
        compiled_model = model.compile(
            loss="mean_squared_error", 
            metrics=[f1],
            optimizer=tf.keras.optimizers.Adam(), # по умолчанию learning rate=10e-3
        )
        return compiled_model
    

class LSTM():
    """Класс создаёт модель LSTM.
    Параметры:
    ----------
    n_timesteps (_int_) - кол-во временных периодов
    n_channels (_int_) - кол-во датчиков
    output_units - 
    lstm_units - размерность модели
    """
    def __init__(self, X_train_nn, x_trn_pred_dict):
        super(LSTM, self).__init__()
        self.n_timesteps = X_train_nn.shape[1]
        self.n_channels = X_train_nn.shape[2]
        self.output_units = np.mean(x_trn_pred_dict[3], axis=0).shape[-1]  # среднее предсказание от 3-х моделей SRNN
        self.lstm_units = config['lstm_units']
        print(f"input_shape = {(self.n_timesteps, self.n_channels)} | output_units = {self.output_units}")


    def build_model_LSTM(self):
        """Метод формирования модели
        """
        input_channels = x = tf.keras.layers.Input(shape=(self.n_timesteps, self.n_channels))

        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.LSTM(units=self.lstm_units, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(units=self.lstm_units, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(units=self.lstm_units, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(units=self.lstm_units, return_sequences=True, dropout=0.5)(x) 

        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        output = tf.keras.layers.Dense(units=self.output_units, activation='softmax')(x)

        model_lstm = tf.keras.Model(
            inputs=input_channels,
            outputs=output,
            name="model_LSTM"
        )        
        print(model_lstm.summary())
        
        return model_lstm