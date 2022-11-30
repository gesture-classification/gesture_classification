# Файл для хранения функций третьего этапа соревнований от Моторики

# Импортируем библиотеки
import pandas as pd
import numpy as np

# графические библиотеки
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

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


gestures = ['"open"',  # 0
            '"пистолет"',  # 1
            'сгиб большого пальца',  # 2
            '"ok"',  # 3
            '"grab"',  # 4
            '"битые" данные',  # -1
]


def plot_history(history):
    
    """
    Функция визуализации процесса обучения модели       
    """
    f1_sc = history.history['f1']  
    loss = history.history['loss']

    f1_sc_val = history.history['val_f1'] # на валидационной выборке
    val_loss = history.history['val_loss']

    epochs = range(len(f1_sc))

    # визуализация систем координат
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 4))

    ax[0].plot(epochs, loss, 'b', label='Training loss')
    ax[0].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[0].set_title('Качество обучения модели')
    ax[0].legend()

    ax[1].plot(epochs, f1_sc, 'b', label='Training f1_score')
    ax[1].plot(epochs, f1_sc_val, 'r', label='Training val_f1_score')
    ax[1].set_title('Изменение f1_score') # изменение f1-score
    ax[1].legend()

    #fig.show()


def f1(y_true, y_pred):
        
    def recall(y_true, y_pred):
        """
        Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        see: https://stackoverflow.com/questions/66554207/calculating-micro-f-1-score-in-keras
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
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
        true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives/(predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision + recall + K.epsilon()))


# Callbacks that used for training model
def callbacks(lr, num_train, PATH_BEST_MODEL='logs_and_figures'):
    
    checkpoint = ModelCheckpoint(
        os.path.join(PATH_BEST_MODEL, 'best_model_rnn_' + str(num_train) + '.hdf5'), 
        monitor='val_f1', 
        verbose=1, 
        mode='max', 
        save_best_only=True
    )

    earlystop = EarlyStopping(
        monitor='val_f1', 
        mode='max', 
        patience=150, 
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_f1', 
        mode='max', 
        factor=0.9, 
        patience=15, # можно 10
        verbose=1, 
        min_lr=lr/10000
    )
    
    return [checkpoint, earlystop, reduce_lr]