import numpy as np
import tensorflow as tf
from keras.models import Model

# Импорт параметров
# from utils.functions import config_reader

# config = config_reader()


class ModelSimpleRNN(Model):
    """Класс создаёт модель SimpleRNN, наследуя класс от tf.keras.
    Параметры:
    ----------
    n_timesteps (_int_) - кол-во временных периодов
    n_channels (_int_) - кол-во датчиков
    output_units - 
    units - размерность модели из конфига 
    """    
    def __init__(self, X_train_nn: np.ndarray, y_train_nn: np.ndarray, units: int):

        super().__init__()
        # ------- параметры ------------
        self.n_timesteps = None
        self.n_channels = X_train_nn.shape[2]
        self.output_units = y_train_nn.shape[-1]
        self.units = units
                
        # -------- слои модели ----------------
        self.input_layer = x = tf.keras.layers.Input(shape=(self.n_timesteps, self.n_channels))
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.SimpleRNN(units=self.units, return_sequences=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        self.output_layer = tf.keras.layers.Dense(units=self.output_units, activation='sigmoid')(x)
        
        print(f"input_shape = {(self.n_timesteps, self.n_channels)} | output_units = {self.output_units}")

    def build_model(self):
        """Метод формирования модели. 
        """
        model = tf.keras.Model(
            inputs=self.input_layer,
            outputs=self.output_layer,
            name="model_SRNN"
        ) 
        model.summary()
        return model


class ModelLSTM(Model):
    """Класс создаёт модель LSTM, наследуя класс от tf.keras.
    Параметры:
    ----------
    n_timesteps (_int_) - кол-во временных периодов
    n_channels (_int_) - кол-во датчиков
    output_units - конечный слой модели
    lstm_units - размерность модели из конфига
    """
    def __init__(self, X_train_nn: np.ndarray, x_trn_pred_dict: np.ndarray, lstm_units: int):
        
        super().__init__()
        # ------- параметры ------------
        self.n_timesteps = None
        self.n_channels = X_train_nn.shape[2]
        self.output_units = np.mean(x_trn_pred_dict[3], axis=0).shape[-1]  # среднее предсказание от 3-х моделей SRNN
        self.lstm_units = lstm_units
        
        # -------- слои модели ----------------
        self.input_channels = x = tf.keras.layers.Input(shape=(self.n_timesteps, self.n_channels))
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(units=self.lstm_units, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(units=self.lstm_units, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(units=self.lstm_units, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(units=self.lstm_units, return_sequences=True, dropout=0.5)(x) 
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        self.output_channels = tf.keras.layers.Dense(units=self.output_units, activation='softmax')(x)
        
        print(f"input_shape = {(self.n_timesteps, self.n_channels)} | output_units = {self.output_units}")

    def build_model(self):
        """Метод формирования модели
        """
        model = tf.keras.Model(
            inputs=self.input_channels,
            outputs=self.output_channels,
            name="model_LSTM"
        )        
        model.summary()
        return model
    