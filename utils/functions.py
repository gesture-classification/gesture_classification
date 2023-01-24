# Импортируем библиотеки
import re
import os
import sys
import glob
import json
from dotmap import DotMap
import numpy as np

from utils import one_learning
from utils.data_reader import DataReader
from utils.inference import MakeInference


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


def get_id_from_data():
    """Функция загрузки номеров пилотов из данных в папке data

    Returns:
        id_pilot_numb_list (_int_): список с номерами пилотов
    """    
    id_pilot_numb_list = [] 
    pattern = r'\d+'
    pattern_2 = 'y_train_'

    X_train_list = glob.glob('data/X_train_*.npy')
    files_list = os.listdir('data')
    
    for item in X_train_list:
        id_pilot_num = re.search(pattern, item)[0]
        if pattern_2 + id_pilot_num + '.npy' in files_list:
            id_pilot_numb_list.append(int(id_pilot_num))
    
    return id_pilot_numb_list


def choice_id_pilot(id_request: str) -> int:
    """Функция выбора номера пилота для обучения модели

    Args:
        id_request (_str_): текст запроса на выбор номера пилота согласно опции

    Returns:
        id_pilot (_int_): номер пилота
    """    
    
    id_pilots_list = get_id_from_data()
    print(id_request)
    id_pilot = int(str(input(f'Введите номер пилота из списка {id_pilots_list}: ')))

    if id_pilot not in id_pilots_list:#(1, 2, 3):
        id_pilot = int(str(
            input(f'\nВведите номер пилота из списка {id_pilots_list},\nдругой выбор - выйти: ')))

    if id_pilot in id_pilots_list:#(1, 2, 3):
        print('\nПодождите, идет расчет...\n')
    else:
        id_pilot = False

    return id_pilot


def choice_train_inference():
    """Функция выбора: обучение модели или получение инференса для одного пилота.
    В зависимости от выбора пользователя:
    -------------
    1 - predictions.csv - предсказание на тестовых данных
    2 - model_lstm_.h5 - обученная модель
    3 - выход из программы
     
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

            id_pilot = choice_id_pilot(pr)
            if not id_pilot:
                break

            path_to_X_test_dataset = config.PATH[3:] + config.mounts[str(id_pilot)].path_X_test_dataset
            X_test_dataset = DataReader(path_to_X_test_dataset).data

            MakeInference.create_prediction(X_test_dataset, config, id_pilot)

        elif choice == '2':
            pr = '\nВведите номер пилота, по которому загрузить данные X_train и y_train\n' \
                 'для обучения и сохранения обученной модели\n'

            id_pilot = choice_id_pilot(pr)
            if not id_pilot:
                break

            learning_pilot = one_learning.OneLearning(config)
            learning_pilot.save_lstm_model(id_pilot=id_pilot)

def search_five_element(array_predicted, start_counter, y_start, y_end):
    if y_start > y_end:
        start_counter = array_predicted.size
        
        # Условие, когда класс до ступеньки больше, чем класс после
        while start_counter > 0 :
            
            # ищем 5 элементов, неравных нулю
            if y_start == np.mean(array_predicted[start_counter-5:start_counter]):
                return(start_counter)
            else:
                start_counter -= 1
        
    else:
        # Условие, когда класс до ступеньки меньше, чем класс после
        while start_counter < array_predicted.size:
            # ищем 5 элементов, неравных нулю
            if y_end == np.mean(array_predicted[start_counter:start_counter+5]):
                return(start_counter)
            else:
                start_counter += 1
                    
def make_best_step(ii, mounts, Pilot_id, x_trn_pred_dict):
    #mounts[id_pilot]['y_train_nn'][ii].argmax(axis=-1)
    #y_pred_train_nn = mounts[Pilot_id]['y_pred_train_nn']
    
    y_pred_train_nn_mean = np.mean(x_trn_pred_dict[Pilot_id], axis=0)
    
    # Первый и последний элемент y_train
    y_start = mounts[Pilot_id]['y_train_nn'][ii].argmax(axis=-1)[5]
    y_end = mounts[Pilot_id]['y_train_nn'][ii].argmax(axis=-1)[-5]
    
    # Ступенька в y_train_nn
    start_counter = np.nonzero(np.diff(mounts[Pilot_id]['y_train_nn'][ii].argmax(axis=-1)))[0][0] + 1
    
    # Смотрим тестовый массив
    array_predicted = y_pred_train_nn_mean[ii].argmax(axis=-1)
                
    predicted_step = search_five_element(array_predicted=array_predicted, start_counter=start_counter, y_start=y_start, y_end=y_end)

    # Обрабатываем ступеньку
    best_step = np.zeros(array_predicted.size, np.int16)
    best_step[0:predicted_step] = y_start
    best_step[predicted_step:] = y_end
    
    return best_step