# Импортируем библиотеки
import os
import sys
import json
import random
from dotmap import DotMap
import numpy as np 

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#from utils import one_learning
from utils.data_reader import DataReader
#from utils.inference import MakeInference


# библиотека взаимодействия с интерпретатором
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def config_reader(path_to_json_conf: str) -> dict:
    """Функция загрузки параметров конфигурации в память.

    Args:
    ------------
    path_to_json_conf (_str_): путь к файлу конфигурации

    Returns:
    ------------
    config (dict): словарь с параметрами конфигурации
    """    
    with open(path_to_json_conf, 'r') as config_file:
        config_dict = json.load(config_file)
    
    config = DotMap(config_dict)
    
    return config


def f1(y_true, y_pred):
    """Функция для расчета метрики f1_score, Precision, Recall
    
    Args:
        y_true (int): исходные данные в диапазоне [0, 1]
        y_pred (int): предсказанные данные в диапазоне [0, 1]

    Returns:
        recall_res (np.float64): метрика f1-score
    """        
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    def recall(y_true, y_pred):
        """
        Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        see: https://stackoverflow.com/questions/66554207/calculating-micro-f-1-score-in-keras
        """
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_res = true_positives / (possible_positives + K.epsilon())
        return recall_res

    def precision(y_true, y_pred):
        """
        Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_res = true_positives/(predicted_positives + K.epsilon())
        return precision_res

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall)/(precision + recall + K.epsilon()))


def callbacks(
    num_train, PATH_BEST_MODEL, monitor, verbose, mode, save_best_only,  #  for checkpoint
    stop_patience, restore_best_weights,  # for earlystop
    factor, min_lr, reduce_patience,      # for reduce_lr
    ):
    """Функция управления этапами обучения модели

    Args:
        min_lr (_float_): нижняя граница learning rate, по которой обучение прекращается
        num_train (_int_): номер пилота
        monitor (str) - значение метрики 
        mode (str)- режим работы функции {"auto", "min", "max"}. Max - остановка обучения, когда метрика не увеличивается; 
        reduce_patience (_int_): количество эпох, после которого learning rate снижается в случае, если метрика не улучшается.
        stop_patience (_int_):  количество эпох, после которого обучение останавливается, если метрика не улучшается.
        PATH_BEST_MODEL (_str_): путь сохранения лучшей модели (из конфига).
        save_best_only (bool): Если True, то сохраняет только модели с лучшим скором.
    """      
    
    # сохранение лучшей модели
    checkpoint = ModelCheckpoint(
        os.path.join(PATH_BEST_MODEL, 'best_model_rnn_' + str(num_train) + '.hdf5'), 
        monitor=monitor, 
        verbose=verbose, 
        mode=mode, 
        save_best_only=save_best_only
    )

    # остановка обучения при отсутствии улучшения заданной метрики
    earlystop = EarlyStopping(
        monitor=monitor, 
        mode=mode, 
        patience=stop_patience, 
        restore_best_weights=restore_best_weights
    )

    # снижение learning rate при отсутствии улучшения заданной метрики 
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor, 
        mode=mode,  
        factor=factor, 
        patience=reduce_patience,  # можно 10
        verbose=verbose, 
        min_lr=min_lr
    )
    
    return [checkpoint, earlystop, reduce_lr]


def reset_random_seeds(seed_value):
    """Функция задания seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)


def main_id_pilot(pr):
    """Функция выбора номера пилота для обучения модели

    Args:
        pr (_int_): номер пилота

    Returns:
        id_pilot (_int_): номер пилота
    """    
    
    print(pr)
    id_pilot = int(str(input('Введите 1, 2 или 3: ')))

    if id_pilot not in (1, 2, 3):
        id_pilot = int(str(input('\nВведите номер пилота 1, 2 или 3,\nдругой выбор - выйти: ')))

    if id_pilot in (1, 2, 3):
        print('\nПодождите, идет расчет...\n')
    else:
        id_pilot = False

    return id_pilot


def main():
    
    """Description
     
    """    
    config = {} 
    while True:
        print('Выберите нужную опцию:\n'
              '     1 - сделать предсказание для тестовых данных на основании предобученной модели,\n'
              '     2 - обучить новую модель для произвольного пилота,\n'
              '     3 - выйти.\n')

        choice = str(input('Введите 1, 2 или 3: '))
        if choice == '3':
            break
        elif choice not in ('1', '2', '3'):
            continue

        if not config:
            path_to_config = 'config/data_config.json'
            config = config_reader(path_to_config)

        if choice == '1':
            pr = '\nВведите номер пилота, по которому загрузить данные X_test\n' \
                 'для получения predict с помощью уже обученной на данных этого пилота модели\n'

            id_pilot = main_id_pilot(pr)
            if not id_pilot:
                break

            path_to_X_test_dataset = config.PATH[3:] + config.mounts[str(id_pilot)].path_X_test_dataset
            X_test_dataset = DataReader(path_to_X_test_dataset).data

            # MakeInference.create_prediction(X_test_dataset, config, id_pilot)

        elif choice == '2':
            pr = '\nВведите номер пилота, по которому загрузить данные X_train и y_train\n' \
                 'для обучения и сохранения обученной модели\n'

            id_pilot = main_id_pilot(pr)
            if not id_pilot:
                break

            # learning_pilot = one_learning.OneLearning(config)
            # learning_pilot.save_lstm_model(id_pilot=id_pilot)
