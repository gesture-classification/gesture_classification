# Файл для хранения функций построения графиков

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
import time


gestures = ['"open"',  # 0
            '"пистолет"',  # 1
            'сгиб большого пальца',  # 2
            '"ok"',  # 3
            '"grab"',  # 4
            '"битые" данные',  # -1
]


# Словарь для последующей агрегации данных. Изначально прописаны названия файлов в архиве
mounts = {
    1 : {
        'path_X_train' : 'X_train_1.npy',
        'path_y_train' : 'y_train_1.npy',
        'path_X_test_dataset' : 'X_test_dataset_1.pkl',
    },
    2 : {
        'path_X_train' : 'X_train_2.npy',
        'path_y_train' : 'y_train_2.npy',
        'path_X_test_dataset' : 'X_test_dataset_2.pkl',
    },
    3 : {
        'path_X_train' : 'X_train_3.npy',
        'path_y_train' : 'y_train_3.npy',
        'path_X_test_dataset' : 'X_test_dataset_3.pkl',
    }
}


def get_sensor_list(Pilot_id, mounts, print_active=False):
    """ Функция печати и импорта в память всех номеров датчиков.
    
    ---Аргументы функции:-----
    Pilot_id - номер пилота,
    mounts - словарь с данными. 
    """
    X_train=mounts[Pilot_id]['X_train']

    df = pd.DataFrame(
        data = X_train, 
        index = [s for s in range(X_train.shape[0])], 
        columns = [s for s in range(X_train.shape[1])]
    ).T

    
    # Создадим список индексов активных и пассивных датчиков. Среднее значение сигнала не превышает 200 единиц.
    active_sensors, passive_sensors = list(), list()
      
    for i in range(df.shape[0]):
        # если средняя амплитуда превышает 200, то добавляем индекс в список 'active_sensors' (надежных датчиков). 
        if df.iloc[i].mean() > 200:
            active_sensors.append(i)
        
        #Остальные датчики с малой амплитудой - в список ненадёжных.      
        else:
            passive_sensors.append(i)

    if print_active is True:
        print(f"Активные датчики пилота " + str(Pilot_id) + ": ", active_sensors)
        print(f"Пассивные датчики пилота " + str(Pilot_id) + ": ", passive_sensors) 
    
    return active_sensors, passive_sensors 



def get_signals_plot(data, test_id, plot_counter):
    """Функция отображения показаний датчиков наблюдений для каждого пилота

    Args:
        data (dict): _description_
        test_id (list): номер наблюдения
        plot_counter (int): номер рисунка
    """    
    
    for mount_name, mount in mounts.items():
        X_test_dataset = data
        
        fig, axx = plt.subplots(3, 1, sharex=True, figsize=(12, 5))

        test_id = 3  # номер наблюдения 
        plt.sca(axx[0])
        plt.plot(X_test_dataset[test_id].T, lw=0.5)
        plt.title(f'Размер наблюдения: {X_test_dataset[test_id][mount_name].T.shape[0]} временных промежутков')
        
        test_id = 1  # номер наблюдения
        plt.sca(axx[1])
        plt.plot(X_test_dataset[test_id].T, lw=0.5)
        plt.title(f'Размер наблюдения: {X_test_dataset[test_id][mount_name].T.shape[0]} временных промежутков')
        
        test_id = 8  # номер наблюдения
        plt.sca(axx[2])
        plt.plot(X_test_dataset[test_id].T, lw=0.5)
        plt.title(f'Размер наблюдения: {X_test_dataset[test_id][mount_name].T.shape[0]} временных промежутков')
        
        plt.suptitle(f"Рис. {plot_counter} - Сигналы датчиков в наблюдениях  пилота {mount_name}", y=-0.1, fontsize=16)
        plt.tight_layout()
        
        plt.savefig(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png')
        plt.show();

        break



def get_all_sensors_plot(data, Pilot_id, timesteps:list, mounts, plot_counter=1):
    """
    Функция построения диаграммы показаний датчиков заданного временного периода. Аргументы функции:
    Pilot_id - номер пилота;
    timesteps - лист из двух временных периодов;
    mounts - словарь с данными;
    plot_counter - порядковый номер рисунка.
    """
    
    X_train=mounts[Pilot_id][data]

    df = pd.DataFrame(
        data = X_train, 
        index = [s for s in range(X_train.shape[0])], 
        columns = [s for s in range(X_train.shape[1])]
    )
    
    fig = go.Figure()
    fig = px.line(data_frame=df.iloc[timesteps[0]:timesteps[1],:])
    
    fig.update_layout(
        title=dict(text=f'Рис. {plot_counter} - сигналы датчиков пилота {Pilot_id}', x=.5, y=0.05, xanchor='center'), 
        xaxis_title_text = 'Время, сек', yaxis_title_text = 'Сигнал датчиков', # yaxis_range = [0, 3000],
        legend_title_text='Индекс <br>датчика',
        width=600, height=400,
        margin=dict(l=100, r=60, t=80, b=100),
    )

    #fig.show()

    # сохраним результат в папке figures. Если такой папки нет, то создадим её
    if not os.path.exists("/gesture_classification/logs_and_figures"):
        os.mkdir("/gesture_classification/logs_and_figures")
    
    fig.write_image(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png') #, engine="kaleido"


def get_active_passive_sensors_plot(Pilot_id, timesteps:list, mounts, plot_counter=1):
    """
    Функция построения графика показаний активных и пассивных датчиков. Аргументы функции:
    Pilot_id - номер пилота;
    timesteps - лист из двух временных периодов;
    mounts - словарь с данными;
    plot_counter - порядковый номер рисунка.  
    """
    #get_sensor_list()
    
    X_train=mounts[Pilot_id]['X_train']

    df = pd.DataFrame(
        data = X_train, 
        index = [s for s in range(X_train.shape[0])], 
        columns = [s for s in range(X_train.shape[1])]
    )

    # списки сенсоров
    active_sensors, passive_sensors = get_sensor_list(Pilot_id, mounts)  #, print_active=True

    #timesteps=[time_start, time_end]

    df = pd.DataFrame(
        data = X_train, 
        index = [s for s in range(X_train.shape[0])], 
        columns = [s for s in range(X_train.shape[1])]
    ).iloc[timesteps[0]:timesteps[1],:]
    
        
    df_1 = pd.DataFrame(df[active_sensors], columns=active_sensors)
    df_2 = pd.DataFrame(df[passive_sensors], columns=passive_sensors)

   
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('активные датчики', 'пассивные датчики')
    )
    
    for i in df_1.columns: fig.add_trace(go.Scatter(x=df_1.index, y=df_1[i], name=str(df[i].name)), row=1, col=1)

    for i in df_2.columns: fig.add_trace(go.Scatter(x=df_2.index, y=df_2[i], name=str(df[i].name)), row=1, col=2)

    fig.update_layout(title={'text':f'Рис. {plot_counter} - Активные и пассивные датчики пилота {Pilot_id} в период {timesteps[0],timesteps[1]}', 'x':0.5, 'y':0.05}
    )

    fig.update_layout(width=1000, height=400, legend_title_text ='Номер датчика',
                        xaxis_title_text  = 'Время',  yaxis_title_text = 'Сигнал датчика', yaxis_range=  [0, 4000], 
                        xaxis2_title_text = 'Время', yaxis2_title_text = 'Сигнал датчика', yaxis2_range= [0 , 200],
                        margin=dict(l=100, r=60, t=80, b=100), 
                        #showlegend=False # легенда загромождает картинку
    )

    #fig.show()

    fig.write_image(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png') #, engine="kaleido"
    

def plot_history(history, plot_counter):
    
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
    ax[1].set_title(f"Изменение f1_score") # изменение f1-score
    ax[1].legend()

    fig.suptitle(f"Рис. {plot_counter} - Ход обучения модели", y=-0.1, fontsize=14)
              
    fig.savefig(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png')
    fig.show();


def get_gesture_prediction_plot(Pilot_id, i, y_pred_train_nn_mean, mounts, plot_counter):
    
    """Функция построения графиков: сигнал датчиков оптомиографии, изменение класса жеста, вероятности появления жеста и предсказание класса жеста.
    
    ----Агументы:----
    Pilot_id = 3  # номер пилота
    plot_counter = 1 # номер рисунка
    i - номер наблюдениия
    mounts - словарь с данными
    """
    
    mount = mounts[Pilot_id]         # выбираем номер пилота
    X_train_nn = mount['X_train_nn']
    y_train_nn = mount['y_train_nn']
    #y_pred_train_nn = mount['y_pred_train_nn']
    #y_pred_train_nn_mean = np.mean(x_trn_pred_dict[Pilot_id], axis=0)

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
    plt.suptitle(f'Рис. {plot_counter} - наблюдение {i} пилота {Pilot_id}' , y=-0.01, fontsize=16)
    
    plt.subplots_adjust(  left=0.1,   right=0.9,
                        bottom=0.1,     top=0.9,
                        wspace=0.1,  hspace=0.4)
 
    ax[0].plot(X_train_nn[i])
    ax[0].set_title('Сигналы датчиков ОМГ')

    ax[1].imshow(y_train_nn[i].T, origin="lower")
    ax[1].set_aspect('auto')
    ax[1].set_title('Класс / жест')
    ax[1].set_yticks(
        np.arange(5),
        ['Open', 'Pistol', 'Thumb', 'OK', 'Grab']
    )

    ax[2].imshow(y_pred_train_nn_mean[i].T, origin="lower")
    ax[2].set_aspect('auto')
    ax[2].set_title('Предсказание вероятностей появления классов жестов')
    ax[2].set_yticks(
        np.arange(5),
        ['Open', 'Pistol', 'Thumb', 'OK', 'Grab']
    )

    ax[3].plot(y_pred_train_nn_mean[i].argmax(axis=-1))
    ax[3].set_aspect('auto')
    ax[3].set_title('Предсказание классов жестов')
    ax[3].set_yticks(
        np.arange(5),
        ['Open', 'Pistol', 'Thumb', 'OK', 'Grab']
    )
    ax[3].set_xlabel('Время')
    plt.tight_layout()



def get_signal_and_train_plots(Pilot_id, timesteps:list, sensors:list, mounts, plot_counter=1):
    """ Функция построения графиков: сигнал датчиков оптомиографии, изменение класса жеста. 
    
    -----Агументы:-------------
    Pilot_id - номер пилота;
    timesteps - список из двух временных периодов;
    sensors - список датчиков;
    mounts - словарь с данными;
    plot_counter - номер рисунка.
    """
    X_train=mounts[Pilot_id]['X_train']
    y_train=mounts[Pilot_id]['y_train']

    
    df_1 = pd.DataFrame(
        data = X_train, 
        index = [s for s in range(X_train.shape[0])], 
        columns = [s for s in range(X_train.shape[1])]
    ).iloc[timesteps[0]:timesteps[1],:][sensors]


    df_2 = pd.DataFrame(
        data = y_train, 
        index = [s for s in range(y_train.shape[0])]
    ).iloc[timesteps[0]:timesteps[1],:]

    
    fig = make_subplots(rows=2, cols=1, 
        subplot_titles=(f'X_train - сигналы датчиков', 'y_train - сигнал манипулятора'), vertical_spacing = 0.15,
    )

    for i in df_1.columns: 
        fig.add_trace(go.Scatter(x=df_1.index, y=df_1[i], name=str(df_1[i].name)), row=1, col=1)
    

    for i in df_2.columns: 
        fig.add_trace(go.Scatter(x=df_1.index, y=df_2[i], name=str(df_1[i].name)), row=2, col=1)

    
    fig.update_layout(title={'text':f'Рис. {plot_counter} - Cигналы датчиков {sensors} пилота {Pilot_id} и сигнал манипулятора', 
    'x':0.5, 'y':0.01}
    )

    fig.update_layout(width=600, height=600, legend_title_text ='Номер датчика',
                        xaxis_title_text  = 'Время',  yaxis_title_text = 'Сигнал датчика', #yaxis_range=[1500, 1700], 
                        xaxis2_title_text = 'Время', yaxis2_title_text = 'Жест', yaxis2_range= [-1 , 5],
                        yaxis2 = dict(
                                    tickmode='array', #change 1
                                    tickvals = np.arange(5), #change 2
                                    ticktext = ['Open', 'Pistol', 'Thumb', 'OK', 'Grab']), 
                        margin=dict(l=40, r=60, t=30, b=80),  
                        showlegend=False # легенда загромождает картинку
    )
    
    #fig.show()
    
    # сохранение иллюстрации
    fig.write_image(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png') #, engine="kaleido"


def get_signal_derivative_and_normalized_plot(Pilot_id, timesteps:list, sensors:list, mounts, plot_counter=1):

    """Функция построения графиков: сигнал датчиков оптомиографии, изменение класса жеста, вероятности появления жеста и предсказание.
    
    ------Агументы:---------
    Pilot_id - номер пилота;
    timesteps - список из двух временных периодов;
    sensors - список датчиков;
    mounts - словарь с данными;
    plot_counter - номер рисунка.
    """
    X_train=mounts[Pilot_id]['X_train']
        
    df_1 = pd.DataFrame(
        data = X_train, 
        index = [s for s in range(X_train.shape[0])], 
        columns = [s for s in range(X_train.shape[1])]
    ).iloc[timesteps[0]:timesteps[1],:][sensors]

    # Нормализация данных
    scaler = StandardScaler()
    scaler.fit(df_1)
    df_2 = pd.DataFrame(scaler.transform(df_1), index = range(df_1.index[0], df_1.index[-1]+1))
       
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=(
            f'Производная сигналов датчиков', f'Производная нормализованных сигналов датчиков', 
            f'Квадрат производной сигналов датчиков', f'Квадрат производной нормализованных сигналов датчиков'
        ), vertical_spacing = 0.1,
    )


    df_3 = pd.DataFrame(df_1.diff(), index = range(df_1.index[0], df_1.index[-1]+1))

    for i in df_3.columns: 
        fig.add_trace(go.Scatter(x=df_1.index, y=df_3[i], name=str(df_3[i].name)), row=1, col=1)

    #  датасет производной
    df_4 = pd.DataFrame(np.power(df_1.diff(),2), index = range(df_1.index[0], df_1.index[-1]+1))

    for i in df_4.columns: 
        fig.add_trace(go.Scatter(x=df_1.index, y=df_4[i], name=str(df_1[i].name)), row=2, col=1)
   

    for i in df_2.columns: 
        fig.add_trace(go.Scatter(x=df_1.index, y=df_2[i], name=str(df_2[i].name)), row=1, col=2)

    #  датасет квадрата производной нормализованного сигнала
    df_6 = pd.DataFrame(np.power(df_2.diff(),2), index = range(df_2.index[0], df_2.index[-1]+1))

    for i in df_6.columns: 
        fig.add_trace(go.Scatter(x=df_1.index, y=df_6[i], name=str(df_6[i].name)), row=2, col=2)

    fig.update_layout(title={'text':f'Рис. {plot_counter} - Преобразование сигнала датчиков {sensors} пилота {Pilot_id}', 'x':0.5, 'y':0.01}
    )

    fig.update_layout(width=1200, height=800, legend_title_text =f'Номер датчика ',
                        xaxis_title_text  = 'Время',  yaxis_title_text = 'Сигнал датчика', #yaxis_range=[0, 4000], 
                        xaxis2_title_text = 'Время', yaxis2_title_text = 'Нормализованный сигнал датчика', #yaxis2_range= [0 , 8],
                        xaxis3_title_text = 'Время', yaxis3_title_text = 'Сигнал датчика', #yaxis3_range= [-1 , 8],
                        xaxis4_title_text = 'Время', yaxis4_title_text = 'Нормализованный сигнал датчика', #yaxis2_range= [0 , 8],
                        margin=dict(l=40, r=60, t=30, b=80), 
                        showlegend=False # легенда загромождает картинку
    )

    #fig.show()
    
    # сохранение иллюстрации
    fig.write_image(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png') #, engine="kaleido"
    
    
def get_display_data(mounts, plot_counter):
    """Функция отображения тренировочных данных (X_train, y_train)) для всех пилотов в датасете
    ------Агументы:---------
    mounts (dict) - словарь, содержащий словари с данными: X_train, y_train, x_test
    plot_counter (int) - номер диаграммы
    """    
    for mount_name, mount in mounts.items():
        X_train = mount['X_train']
        y_train = mount['y_train']
        
        # выбор моментов, где происходит изменение y_train
        events = np.where(np.abs(np.diff(y_train)) > 0)[0]
        
        fig, axx = plt.subplots(2, 1, sharex=True, figsize=(12, 5))
        plt.sca(axx[0])
        plt.plot(X_train, lw=0.5)
        plt.title(f'X_train #{mount_name}')
        yl = plt.ylim()
        plt.vlines(events, *yl, color='r', lw=0.5, alpha=0.5)
        
        plt.sca(axx[1])
        plt.plot(y_train, lw=0.5)
        plt.title('y_train')
        yl = plt.ylim()
        plt.vlines(events, *yl, color='r', lw=0.5, alpha=0.5)
        plt.yticks(
            np.arange(-1, 5),
            ['Bad data', 'Open', 'Pistol', 'Thumb', 'OK', 'Grab']
        )
        plt.grid(axis='y')
        
        plt.title(f"Y_train #{mount_name}") 
        
        plt.tight_layout()
    
    # plt.show() - не вызывать для корретного логгирования
    fig.suptitle(f"Рис. {plot_counter} - Сигналы датчиков и классы жестов", y=-0.1, fontsize=12);    
    
    plt.savefig(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png')
    