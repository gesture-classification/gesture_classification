import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Layer
from keras.models import Model

# Импорт параметров
from utils.functions import config_reader, callbacks, f1

config = config_reader() #'../config/data_config.json'

class SimpleRNN(Model):  
    """Класс создаёт модель SimpleRNN, наследуя класс от tf.keras.
    Параметры:
    ----------
    n_timesteps (_int_) - кол-во временных периодов
    n_channels (_int_) - кол-во датчиков
    output_units - 
    units - размерность модели из конфига 
    """    
    def __init__(self, X_train_nn, y_train_nn, units=config['simpleRNN_units']):
        
        super(SimpleRNN, self).__init__()
        #------- параметры ------------
        self.n_timesteps = None #X_train_nn.shape[1]
        self.n_channels = X_train_nn.shape[2]
        self.output_units = y_train_nn.shape[-1]
        self.units = units
                
        #-------- слои модели ----------------
        self.input_layer = self.x = tf.keras.layers.Input(shape=(self.n_timesteps, self.n_channels))
        self.layer1 = tf.keras.layers.BatchNormalization()(self.x)
        self.layer2 = tf.keras.layers.SimpleRNN(units=self.units, return_sequences=True)(self.x)
        self.layer3 = tf.keras.layers.BatchNormalization()(self.x)
        self.output_layer = tf.keras.layers.Dense(units=self.output_units, activation='sigmoid')(self.x)
        
        print(f"input_shape = {(self.n_timesteps, self.n_channels)} | output_units = {self.output_units}")
    
        self.compiled_model = self.build_model()
        #self.fit_model = self.fit()     
        
    
    def call(self):
        x = self.input_layer
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x # returns results of Output Layer

        
    def build_model(self):
        """Метод формирования модели
        """
        model = tf.keras.Model(
            inputs=self.input_layer,
            outputs=self.output_layer,
            name="model"
        ) 
        model.summary()
        return model
    
    
    def compile(self):
        self.model.compile(
            loss="mean_squared_error", 
            metrics=[f1],
            optimizer='adam', # tf.keras.optimizers.Adam() по умолчанию learning rate=10e-3
        )
        
    
    def fit(self, model, X_train_nn, y_train_nn, val_splt_coef, num_train):
        fitted_model = model.fit(
            X_train_nn = X_train_nn, 
            y_train_nn = y_train_nn, 
            validation_split=val_splt_coef, # validation_split изменяется в цикле
            epochs=500,
            verbose=1,
            callbacks=callbacks(num_train) # остальные параметры - смотри в functions.py
        )
        return fitted_model
        
    

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