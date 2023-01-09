import os
import csv
import numpy as np
from tensorflow import keras

from utils.functions import f1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MakeInference:
    """Получение инференса
    """    
    def __init__(self):
        super().__init__()

    @staticmethod
    def create_prediction(X_test_dataset, config, id_pilot):
        """ Функция создания и сохранения предсказания в файл

        Args:
            X_test_dataset (_pd.DataFrame_): массив тестовых данных для проверки качества предсказания
            config (_dict_): словарь с параметрами конфигурации
            path_to_models_weights (_str_): путь до весов моделей
        """

        path_to_models_weights = config.PATH_FOR_MODEL[3:] + 'model_lstm_' + str(id_pilot) + '.h5'
        m_lstm = keras.models.load_model(path_to_models_weights, compile=False)
        m_lstm.compile(loss="mean_squared_error", metrics=[f1], optimizer=keras.optimizers.Adam())

        y_pred_test_lstm = []

        for i in range(len(X_test_dataset)):
            X_test_i = np.expand_dims(X_test_dataset[i], axis=0).swapaxes(1, 2).astype(np.float64)
            y_pred_test_lstm += [m_lstm.predict(X_test_i, verbose=0)]
        
        y_pred_test_lstm = [arr.argmax(axis=-1) for arr in y_pred_test_lstm]
        
        # Сохранение предсказания в файл
        with open("predictions.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(y_pred_test_lstm)
        print('X_test_dataset predicted!')
        print('Данные сохранены в predictions.csv')
        print('__________________________________\n\n')
        # return y_pred_test_lstm
