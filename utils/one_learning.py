# Установка версий программных пакетов в терминале
# pip install -qr ../requirements.txt

# Импортируем библиотеки
import numpy as np
import sys
import os

# библиотеки машинного обучения
import tensorflow as tf
import random

# Логгирование процесса
# from comet_ml import Experiment

# Библиотека вызова функций, специально разработанных для данного ноутбука
from utils.functions import f1, callbacks, reset_random_seeds
from utils.data_reader import DataReader
from utils.data_loader import DataLoader

# Импортируем модели
from models.models import ModelSimpleRNN, ModelLSTM

# Зафиксируем PYTHONHASHSEED для воспроиизводимости результатов обучения модели
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)

# библиотека взаимодействия с интерпретатором
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

sys.path.insert(1, './')


class OneLearning:
    """
    Обучение модели для одного пилота
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def save_lstm_model(self, id_pilot: int):
        ### Params initializations
        config = self.config

        # Все исходные файлы размещены в папке data
        PATH = config.PATH_PY
        
        # Папка для сохранения весов лучшей модели при обучении (см. ModelCheckpoint)
        PATH_TEMP_MODEL = config.PATH_TEMP_MODEL[3:]
        if not os.path.exists(PATH_TEMP_MODEL):
            os.mkdir(PATH_TEMP_MODEL)

        # Папка для сохранения обученных моделей для последующего предсказания
        PATH_FOR_MODEL = config.PATH_FOR_MODEL[3:]
        if not os.path.exists(PATH_FOR_MODEL):
            os.mkdir(PATH_FOR_MODEL)

        # Установим начальное значение для генератора случайных чисел в Python
        random.seed(seed_value)
        # Установим начальное значение для генератора случайных чисел в Numpy
        np.random.seed(seed_value)
        # Установим начальное значение для генератора случайных чисел в tensorflow 
        tf.random.set_seed(seed_value)
        # Конфигурация Tenzorflow
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, 
                                                inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), 
                                    config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)

        # Словарь для последующей агрегации данных. Изначально прописаны названия файлов в архиве
        mounts = config.mounts.toDict()
        print('Imports Done!', sep='\n\n')

        ### Read Data
        i = str(id_pilot)
        mounts[int(i)] = mounts.pop(i)
        i = int(i)

        # Загружаем данные в словарь с помощью DataLoader
        data_for_nn = DataLoader(i, config, is_notebook_train=False)
        mounts[i]['X_train'] = data_for_nn.X_train
        mounts[i]['y_train'] = data_for_nn.y_train
        mounts[i]['X_train_nn'] = data_for_nn.X_train_nn
        mounts[i]['y_train_nn'] = data_for_nn.y_train_nn
        mounts[i]['X_test_dataset'] = DataReader(
            os.path.join(config.PATH_PY, config.mounts[str(i)].path_X_test_dataset)).data
        print('Read Data Done!', sep='\n\n')

        ### Load and train SimpleRNN
        X_train_nn = mounts[i]['X_train_nn']
        y_train_nn = mounts[i]['y_train_nn']

        # Создаем пустой лист для накопления данных с последующей корректировкой y_train
        mounts[i]['y_trn_nn_ch_list'] = []

        model = ModelSimpleRNN(X_train_nn, y_train_nn,
                               units=config.simpleRNN_units).build_model()

        for splt_coef in range(10, 100, config.simpleRNN_delta_coef_splt): 
            # кол-во разных тренировок зависит от числа разбиений.
            
            val_splt_coef = splt_coef/100

            tf.keras.backend.clear_session()
            reset_random_seeds(seed_value)  # сброс и задание random seed

            model = tf.keras.models.clone_model(model)
            
            model.compile(
                loss="mean_squared_error", 
                metrics=[f1], 
                optimizer='Adam',  # по умолчанию learning rate=10e-3
            )

            history = model.fit(
                X_train_nn, 
                y_train_nn, 
                validation_split=val_splt_coef, 
                callbacks=callbacks(
                    reduce_patience=config.reduce_patience, 
                    stop_patience=config.stop_patience, 
                    num_train=i,
                    PATH_BEST_MODEL=config.PATH_TEMP_MODEL,
                    monitor=config.ModelCheckpoint.monitor, 
                    verbose=config.ModelCheckpoint.verbose, 
                    mode=config.ModelCheckpoint.mode, 
                    save_best_only=config.ModelCheckpoint.save_best_only,
                    restore_best_weights=config.EarlyStopping.restore_best_weights,
                    factor=config.ReduceLROnPlateau.factor, 
                    min_lr=config.ReduceLROnPlateau.min_lr_coeff),  
                    # остальные параметры - смотри в functions.py
                epochs=config.simpleRNN_epochs,
                verbose=config.simpleRNN_verbose
            )
            
            y_pred_train_nn = model.predict(X_train_nn)
            mounts[i]['y_trn_nn_ch_list'].append(y_pred_train_nn)

        # создаём переменную для записи среднего значения предсказаний
        y_pred_train_nn = np.mean(mounts[i]['y_trn_nn_ch_list'], axis=0).argmax(axis=-1)
        
        # сохраняем y_pred_train_n в словарь
        mounts[i]['y_pred_train_nn'] = tf.keras.utils.to_categorical(y_pred_train_nn)

        print('Load and train SimpleRNN Done!', sep='\n\n')


        ### Load and train LSTM
        y_pred_train_nn = mounts[i]['y_pred_train_nn']

        model_lstm = ModelLSTM(X_train_nn, y_pred_train_nn,
                                lstm_units=config.lstm_units).build_model()

        tf.keras.backend.clear_session()
        reset_random_seeds(seed_value)    

        model_lstm.compile(
            loss="categorical_crossentropy", 
            metrics=[f1], 
            optimizer='Adam',  # по умолчанию learning rate=10e-3
        )

        history_lstm = model_lstm.fit(
            X_train_nn,
            y_pred_train_nn,
            validation_split=config.lstm_val_splt,
            epochs=config.lstm_epochs,      
            verbose=config.lstm_verbose,
            callbacks=callbacks(
                    num_train=i,
                    reduce_patience=config.reduce_patience, 
                    stop_patience=config.stop_patience, 
                    PATH_BEST_MODEL=config.PATH_TEMP_MODEL,
                    monitor=config.ModelCheckpoint.monitor, 
                    verbose=config.ModelCheckpoint.verbose, 
                    mode=config.ModelCheckpoint.mode, 
                    save_best_only=config.ModelCheckpoint.save_best_only,
                    restore_best_weights=config.EarlyStopping.restore_best_weights,
                    factor=config.ReduceLROnPlateau.factor, 
                    min_lr=config.ReduceLROnPlateau.min_lr_coeff
                )  # остальные параметры - смотри в functions.py
            )

        mounts[i]['history_lstm'] = history_lstm

        mounts[i]['model_lstm'] = model_lstm

        print('Load and train LSTM model Done!')

        # сохранение обученной модели в папке по пути PATH_FOR_MODEL
        model_lstm.save(os.path.join(PATH_FOR_MODEL, 'model_lstm_' + str(i) + '.h5'), 
                        save_format='h5')

        print('Model LSTM saved!')
        print(f"Данные сохранены в {os.path.join(PATH_FOR_MODEL, 'model_lstm_' + str(i) + '.h5')}")
        print('__________________________________\n\n')
