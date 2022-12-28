# Импортируем библиотеки
import os, sys, json, random
from dotmap import DotMap
import numpy as np 

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# библиотека взаимодействия с интерпретатором
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    


def config_reader(path_to_json_conf:str)->dict:
    """Функция загрузки параметров конфигурации в память.

    Args:
    ------------
    path_to_json_conf (_str_): путь к файлу конфигурации

    Returns:
    ------------
    config (dict): словарь с параметрами кинфигурации
    """    
    with open(path_to_json_conf, 'r') as config_file:
        config_dict = json.load(config_file)
    
    config = DotMap(config_dict)
    
    return config

# Задаём переменную, которой присвоим словарь с переменными конфигурации
#config = config_reader() 

def f1(y_true, y_pred):
    """Функция для расчета метрики f1_score, Precision, Recall

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
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
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """
        Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives/(predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision + recall + K.epsilon()))

 # Функция Callbacks, используемая при обучении модели, включающая
 # checkpoint - сохранение лучшей модели

def callbacks(
    num_train, PATH_BEST_MODEL, monitor, verbose, mode, save_best_only, #  checkpoint
    stop_patience, restore_best_weights,  # earlystop 
    factor, min_lr, reduce_patience,      # reduce_lr
    ):
    """Описание функции

    Args:
        lr (_type_): _description_
        num_train (_type_): _description_
        reduce_patience (_type_, optional): _description_. Defaults to config['reduce_patience'].
        stop_patience (_type_, optional): _description_. Defaults to config["stop_patience"].
        PATH_BEST_MODEL (_type_, optional): _description_. Defaults to config["PATH_BEST_MODEL"].

    Returns:
        _type_: _description_
    """      
          
    checkpoint = ModelCheckpoint(
        os.path.join(PATH_BEST_MODEL, 'best_model_rnn_' + str(num_train) + '.hdf5'), 
        monitor=monitor, 
        verbose=verbose, 
        mode=mode, 
        save_best_only=save_best_only
    )

    earlystop = EarlyStopping(
        monitor=monitor, 
        mode=mode, 
        patience=stop_patience, 
        restore_best_weights=restore_best_weights
    )

    reduce_lr = ReduceLROnPlateau(
        monitor=monitor, 
        mode=mode,  
        factor=factor, 
        patience=reduce_patience, # можно 10
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