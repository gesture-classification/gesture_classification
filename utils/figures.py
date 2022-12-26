# Файл для хранения функций построения графиков

# Импортируем библиотеки
import pandas as pd
import numpy as np

# графические библиотеки
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# библиотеки машинного обучения
from sklearn.preprocessing import StandardScaler

# отображать по умолчанию длину Датафрейма
pd.set_option("display.max_rows", 9, "display.max_columns", 9)

# библиотека взаимодействия с интерпретатором
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# Загрузка констант из файла конфигурации
from utils.functions import config_reader
config = config_reader() 
 

# Словарь c названиями файлов в архиве для агрегации данных 
mounts = config['mounts']

def get_dataframe(Pilot_id:int, mounts:dict, data:np.array='X_train')-> pd.DataFrame: 
    """ Функция создания датафрейма из данных словаря
    Аргументы:
    ---------
    Pilot_id (_int_)- номер пилот
    mounts - словарь с данными, из которого они читаются
    data - ключ словаря
    
    Returns:
    ---------
    numpy: np.array
    """
    data = mounts[Pilot_id][data]
    
    df = pd.DataFrame(
        data = data, 
        index = [s for s in range(data.shape[0])], 
        columns = [s for s in range(data.shape[1])]
    )
    
    return df

def get_sensor_list(Pilot_id:int, mounts:dict, print_active=False):
    """ Функция печати и импорта в память всех номеров датчиков.
    
    ---Аргументы функции:-----
    Pilot_id - номер пилота,
    mounts - словарь с данными,
    print_active - печать активных жестов. 
    """
    
    df = get_dataframe(Pilot_id, mounts=mounts).T

    # Создадим список индексов активных и пассивных датчиков. Среднее значение сигнала не превышает 200 единиц.
    active_sensors, passive_sensors = list(), list()
      
    for i in range(df.shape[0]):
        # если средняя амплитуда превышает 200, то добавляем индекс в список 'active_sensors' (надежных датчиков). 
        if df.iloc[i].mean() > config['level_boundary']:
            active_sensors.append(i)
        
        #Остальные датчики с малой амплитудой - в список ненадёжных.      
        else:
            passive_sensors.append(i)

    if print_active is True:
        print(f"Active sensors of pilot " + str(Pilot_id) + ": ", active_sensors)
        print(f"Passive sensors of pilot " + str(Pilot_id) + ": ", passive_sensors) 
    
    return active_sensors, passive_sensors 


def get_signals_plot(data, mounts:dict, test_id:list, plot_counter:int):
    """Функция отображения показаний датчиков наблюдений для каждого пилота

    Args:
        data (dict): _description_
        mounts - словарь с данными,
        test_id (list): номер наблюдения
        plot_counter (int): номер рисунка
    """    
    
    fig, axx = plt.subplots(3, 1, sharex=True, figsize=(10, 5))
    
    for mount_name, mount in mounts.items():
                
        for n, id in enumerate(test_id): # n-порядковый номер наблюдения; id - индекс наблюдения
    
            plt.sca(axx[n])
            plt.plot(data[id].T, lw=0.5)
            plt.title(f'Test duration: {data[id][mount_name].T.shape[0]} periods') #Размер наблюдения..временных промежутков
        
        fig.suptitle(f"Fig.{plot_counter} - Sensor signals of pilot #{mount_name}", y=-0.1, fontsize=14) # Сигналы датчиков в наблюдениях  пилота
        fig.tight_layout()
        
        plt.savefig(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png')
        #plt.show(); #- не вызывать для корретного логгирования

        break


def get_all_sensors_plot(Pilot_id, timesteps:list, mounts:dict, plot_counter=1):
    """
    Функция построения диаграммы показаний датчиков заданного временного периода. Аргументы функции:
    Pilot_id - номер пилота;
    timesteps - лист из двух временных периодов;
    mounts - словарь с данными;
    plot_counter - порядковый номер рисунка.
    """
    
    df = get_dataframe(Pilot_id, mounts=mounts)
    
    fig = go.Figure()
    fig = px.line(data_frame=df.iloc[timesteps[0]:timesteps[1],:])
    
    fig.update_layout(
        title=dict(text=f'Fig.{plot_counter} - sensor signals of pilot #{Pilot_id}', x=.5, y=0.05, xanchor='center'),  #сигналы датчиков пилота
        xaxis_title_text = 'Periods', yaxis_title_text = 'Sensors signals', # yaxis_range = [0, 3000], Время, сек
        legend_title_text='Sensor <br> index',
        width=600, height=400,
        margin=dict(l=100, r=60, t=80, b=100),
    )

    #fig.show()
   
    fig.write_image(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png') #, engine="kaleido"


def get_active_passive_sensors_plot(Pilot_id:int, timesteps:list, mounts:dict, plot_counter=1):
    """    Функция построения графика показаний активных и пассивных датчиков. Аргументы функции:
    Pilot_id - номер пилота;
    timesteps - лист из двух временных периодов;
    mounts - словарь с данными;
    plot_counter - порядковый номер рисунка.  
    """
      
    # списки сенсоров
    active_sensors, passive_sensors = get_sensor_list(Pilot_id=Pilot_id, mounts=mounts)

    df = get_dataframe(Pilot_id, mounts=mounts).iloc[timesteps[0]:timesteps[1],:]
    
        
    df_1 = pd.DataFrame(df[active_sensors], columns=active_sensors)
    df_2 = pd.DataFrame(df[passive_sensors], columns=passive_sensors)

   
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('active sensors', 'passive sensors') #'активные датчики', 'пассивные датчики'
    )
    
    for i in df_1.columns: fig.add_trace(go.Scatter(x=df_1.index, y=df_1[i], name=str(df[i].name)), row=1, col=1)

    for i in df_2.columns: fig.add_trace(go.Scatter(x=df_2.index, y=df_2[i], name=str(df[i].name)), row=1, col=2)

    fig.update_layout(title={'text':f'Fig.{plot_counter} - Active and passive sensors of pilot #{Pilot_id} within the period {timesteps[0],timesteps[1]}', 'x':0.5, 'y':0.05}
    ) #Активные и пассивные датчики пилота .. в период

    fig.update_layout(width=1000, height=400, legend_title_text ='Номер датчика',
                        xaxis_title_text  = 'Period', # Время
                        yaxis_title_text = 'Sensor signal', #Сигнал датчика
                        yaxis_range=  [0, 4000], 
                        xaxis2_title_text = 'Period', yaxis2_title_text = 'Sensor signal', yaxis2_range= [0 , 200],
                        margin=dict(l=100, r=60, t=80, b=100), 
                        #showlegend=False # легенда загромождает картинку
    )

    #fig.show()

    fig.write_image(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png') #, engine="kaleido"
    

def plot_history(history, plot_counter:int):
    
    """Функция визуализации процесса обучения модели.
    Аргументы:
    history - история обучения модели,
    plot_counter - порядковый номер рисунка.      
    """
    f1_sc = history.history['f1']  
    loss = history.history['loss']

    f1_sc_val = history.history['val_f1'] # на валидационной выборке
    val_loss = history.history['val_loss']

    epochs = range(len(f1_sc))

    # визуализация систем координат
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 3.5))

    ax[0].plot(epochs, loss, 'b', label='Training loss')
    ax[0].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[0].set_xlabel('Epoch', size=11)
    ax[0].set_ylabel('Loss', size=11)
    ax[0].set_title('Training and validation loss') # Качество обучения модели
    ax[0].legend()

    ax[1].plot(epochs, f1_sc, 'b', label='Training f1_score')
    ax[1].plot(epochs, f1_sc_val, 'r', label='Training val_f1_score')
    ax[1].set_xlabel('Epoch', size=11)
    ax[1].set_ylabel('F1-score', size=11)
    ax[1].set_title(f"F-1 score") # изменение f1-score
    ax[1].legend()

    fig.suptitle(f"Fig.{plot_counter} - Model learning", y=-0.1, fontsize=14) #Ход обучения модели
              
    fig.savefig(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png')
    #fig.show(); #- не вызывать для корретного логгирования


def get_gesture_prediction_plot(Pilot_id:int, i:int, y_pred_train_nn_mean, mounts:dict, plot_counter:int):
    
    """Функция построения графиков: сигнал датчиков оптомиографии, изменение класса жеста, вероятности появления жеста и предсказание класса жеста.
    
    ----Агументы:----
    Pilot_id = 3  # номер пилота
    plot_counter = 1 # номер рисунка
    i - номер наблюдениия
    mounts - словарь с данными
    """
    
    X_train_nn = mounts[Pilot_id]['X_train_nn']
    y_train_nn = mounts[Pilot_id]['y_train_nn']
    
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
    plt.suptitle(f'Fig.{plot_counter} - test #{i} of pilot {Pilot_id}' , y=-0.01, fontsize=14) #наблюдение
    
    plt.subplots_adjust(  left=0.1,   right=0.9,
                        bottom=0.1,     top=0.9,
                        wspace=0.1,  hspace=0.4)
 
    ax[0].plot(X_train_nn[i])
    ax[0].set_title('OMG sensors signals') #Сигналы датчиков ОМГ

    ax[1].imshow(y_train_nn[i].T, origin="lower")
    ax[1].set_aspect('auto')
    ax[1].set_title('Gesture class') #Класс жеста манипулятора
    ax[1].set_yticks(
        np.arange(5),  ['Open', 'Pistol', 'Thumb', 'OK', 'Grab']
    )

    ax[2].imshow(y_pred_train_nn_mean[i].T, origin="lower")
    ax[2].set_aspect('auto')
    ax[2].set_title('Predicted probability of gesture class') #Предсказание вероятностей появления классов жестов
    ax[2].set_yticks(
        np.arange(5), ['Open', 'Pistol', 'Thumb', 'OK', 'Grab']
    )

    ax[3].plot(y_pred_train_nn_mean[i].argmax(axis=-1))
    ax[3].set_aspect('auto')
    ax[3].set_title('Predicted gesture class')
    ax[3].set_yticks(
        np.arange(5), ['Open', 'Pistol', 'Thumb', 'OK', 'Grab']
    )
    ax[3].set_xlabel('Period') #Время
    plt.tight_layout()
    
    plt.savefig(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png')


def get_signal_and_train_plots(Pilot_id, timesteps:list, sensors:list, mounts:dict, plot_counter:int=1):
    """ Функция построения графиков: сигнал датчиков оптомиографии, изменение класса жеста. 
    
    -----Агументы:-------------
    Pilot_id - номер пилота;
    timesteps - список из двух временных периодов;
    sensors - список датчиков;
    mounts - словарь с данными;
    plot_counter - номер рисунка.
    """
    df_1 = get_dataframe(Pilot_id, mounts=mounts).iloc[timesteps[0]:timesteps[1],:][sensors]
    
    y_train=mounts[Pilot_id]['y_train']
    
    df_2 = pd.DataFrame(
            data = y_train, 
            index = [s for s in range(y_train.shape[0])]
        ).iloc[timesteps[0]:timesteps[1],:]

    
    fig = make_subplots(rows=2, cols=1, 
        subplot_titles=(f'X_train - sensors signals', 'y_train - original manipulator command'), vertical_spacing = 0.15,
    )

    for i in df_1.columns: 
        fig.add_trace(go.Scatter(x=df_1.index, y=df_1[i], name=str(df_1[i].name)), row=1, col=1)
    

    for i in df_2.columns: 
        fig.add_trace(go.Scatter(x=df_1.index, y=df_2[i], name=str(df_1[i].name)), row=2, col=1)

    
    fig.update_layout(title={'text':f'Fig. {plot_counter} - Sensors #{sensors} signals of the pilot #{Pilot_id} and original manipulator command', 
    'x':0.5, 'y':0.01}
    )

    fig.update_layout(width=600, height=600, legend_title_text ='Sensor id',
                        xaxis_title_text  = 'Period',  yaxis_title_text = 'Sensor signal', #yaxis_range=[1500, 1700], 
                        xaxis2_title_text = 'Period', yaxis2_title_text = 'Gesture', yaxis2_range= [-1 , 5],
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


def get_signal_derivative_and_normalized_plot(Pilot_id:int, timesteps:list, sensors:list, mounts:dict, plot_counter:int=1):

    """Функция построения графиков: сигнал датчиков оптомиографии, изменение класса жеста, вероятности появления жеста и предсказание.
    
    ------Агументы:---------
    Pilot_id - номер пилота;
    timesteps - список из двух временных периодов;
    sensors - список датчиков;
    mounts - словарь с данными;
    plot_counter - номер рисунка.
    """
    #X_train=mounts[Pilot_id]['X_train']

    df_1 = get_dataframe(Pilot_id, mounts=mounts).iloc[timesteps[0]:timesteps[1],:][sensors]

    # Нормализация данных
    scaler = StandardScaler()
    scaler.fit(df_1)
    
    df_2 = pd.DataFrame(scaler.transform(df_1), index = range(df_1.index[0], df_1.index[-1]+1))
       
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=(
            f'Derivative of the signal', f'Derivative of normalized signal', 
            f'Squared of signal derivative', f'Squared of normalized signal derivative'
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

    fig.update_layout(title={'text':f'Fig. {plot_counter} - Sensor #{sensors} signal processing of the pilot #{Pilot_id}', 'x':0.5, 'y':0.01} #Преобразование сигнала датчиков..  пилота..
    )

    fig.update_layout(width=1200, height=800, legend_title_text =f'Номер датчика ',
                        xaxis_title_text  = 'Period',  yaxis_title_text = 'Sensor signal', #yaxis_range=[0, 4000], #Сигнал датчика
                        xaxis2_title_text = 'Period', yaxis2_title_text = 'Signal normalized', #yaxis2_range= [0 , 8], #Нормализованный сигнал датчика
                        xaxis3_title_text = 'Period', yaxis3_title_text = 'Sensor signal', #yaxis3_range= [-1 , 8],
                        xaxis4_title_text = 'Period', yaxis4_title_text = 'Signal normalized', #yaxis2_range= [0 , 8],
                        margin=dict(l=40, r=60, t=30, b=80), 
                        showlegend=False # легенда загромождает картинку
    )

    #fig.show()
    
    # сохранение иллюстрации
    fig.write_image(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png') #, engine="kaleido"
    
    
def get_display_data(mounts:dict, plot_counter:int):
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
    
    #plt.show() #- не вызывать для корретного логгирования
    fig.suptitle(f"Fig.{plot_counter} - Sensor signals and gestures classes", y=-0.1, fontsize=12); # Сигналы датчиков и классы жестов
    
    plt.savefig(f'/gesture_classification/logs_and_figures/fig_{plot_counter}.png')
    