# Файл для хранения функций третьего этапа соревнований от Моторики

# Импортируем библиотеки
import pandas as pd
import numpy as np

# графические библиотеки
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# библиотеки машинного обучения
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

