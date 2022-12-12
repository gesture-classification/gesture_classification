# Файл для хранения функций третьего этапа соревнований от Моторики

# Импортируем библиотеки
import pandas as pd
import numpy as np

# графические библиотеки
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from utils.reader_config import read_config

from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#from plotly.subplots import make_subplots



# библиотеки машинного обучения
#from sklearn.metrics import f1_score
#from sklearn.model_selection import StratifiedKFold, cross_validate
#from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import StandardScaler

# отображать по умолчанию длину Датафрейма
pd.set_option("display.max_rows", 9, "display.max_columns", 9)

# библиотека взаимодействия с интерпретатором
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import os

config = read_config('../config/data_config.json')

gestures = config['gestures']

def f1(y_true, y_pred):
    # Функция для расчета метрики f1_score, Precision, Recall
    
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


# Callbacks that used for training model
def callbacks(
    lr,
    num_train, 
    reduce_patience=config['reduce_patience'], 
    stop_patience=config["stop_patience"], 
    PATH_BEST_MODEL=config["PATH_BEST_MODEL"]
):  #
    # Функция Callbacks, используемая при обучении модели, включающая
    # checkpoint - сохранение лучшей модели
        
    
    checkpoint = ModelCheckpoint(
        os.path.join(PATH_BEST_MODEL, 'best_model_rnn_' + str(num_train) + '.hdf5'), 
        monitor=config["ModelCheckpoint"]["monitor"], 
        verbose=config["ModelCheckpoint"]["verbose"], 
        mode=config["ModelCheckpoint"]["mode"], 
        save_best_only=config["ModelCheckpoint"]["save_best_only"]
    )

    earlystop = EarlyStopping(
        monitor=config["EarlyStopping"]["monitor"], 
        mode=config["EarlyStopping"]["mode"], 
        patience=stop_patience, 
        restore_best_weights=config["EarlyStopping"]["restore_best_weights"]
    )

    reduce_lr = ReduceLROnPlateau(
        monitor=config["ReduceLROnPlateau"]["monitor"], 
        mode=config["ReduceLROnPlateau"]["mode"],  
        factor=config["ReduceLROnPlateau"]["factor"], 
        patience=reduce_patience, # можно 10
        verbose=config["ReduceLROnPlateau"]["verbose"], 
        min_lr=lr/config["ReduceLROnPlateau"]["min_lr_coeff"]
    )
    
    return [checkpoint, earlystop, reduce_lr]