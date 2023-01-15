import os
import numpy as np
import random
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


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
