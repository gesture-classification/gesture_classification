import numpy as np
import mne
from tensorflow import keras

from utils.data_reader import DataReader


class DataLoader:
    """Программа загрузки X_train и y_train для подачи в модель.
    Аргументы:
    -------------
    config (_dict_) - словарь с конфигурацией
    X_train (_pd.DataFrame_)- обучающая выборка
    y_train (_pd.DataFrame_)- сигналы манипулятора
    path_X_train (_str_) - ключ словаря mounts для загрузки X_train
    path_y_train (_str_) - ключ словаря mounts для загрузки  y_train
    id_pilot (_int_)- номер пилота
    """    
    def __init__(self, id_pilot: int, config: dict, is_notebook_train=True):
        super().__init__()
        self.config = config
        if is_notebook_train:
            self.X_train = DataReader(
                self.config['PATH'] + self.config['mounts'][str(id_pilot)]['path_X_train']).data
            self.y_train = DataReader(
                self.config['PATH'] + self.config['mounts'][str(id_pilot)]['path_y_train']).data
        else:
            self.X_train = DataReader(
                self.config['PATH_PY'] + self.config['mounts'][str(id_pilot)]['path_X_train']).data
            self.y_train = DataReader(
                self.config['PATH_PY'] + self.config['mounts'][str(id_pilot)]['path_y_train']).data

        self.X_train_nn, self.y_train_nn = self.create_train_nn(self.X_train, self.y_train, self.config)

    @staticmethod
    def create_train_nn(X_train, y_train, config):
        """Объединение в список всех наблюдений пилота с помощью библиотеки mne

        Args:
        -------
            X_train (_pd.DataFrame_)- обучающая выборка
            y_train (_pd.DataFrame_)- сигналы манипулятора
            config (_dict_) - словарь с конфигурацией

        Returns:
            X_train_nn, y_train_nn (_np.ndarray_)- массивы тренировочных данных
        """
        raw = mne.io.RawArray(
            data=X_train.T,
            info=mne.create_info(
                ch_names=list(np.arange(X_train.shape[1]).astype(str)),
                sfreq=config['points']/config['point_to_time'],
                ch_types='eeg'
                )
            )
        
        raw_y = mne.io.RawArray(
            data=y_train.reshape(1,-1),
            info=mne.create_info(
                ch_names=['y'],
                sfreq=config['points']/config['point_to_time'],
                ch_types='misc'
                )
            )
        raw = raw.add_channels([raw_y])

        # выбор индексов, где происходит изменение жеста
        events = np.where(np.abs(np.diff(y_train)) > 0)[0]

        events = np.stack([
            events,
            np.zeros_like(events),
            np.zeros_like(events)
        ], axis=1)
        
        epochs = mne.Epochs(
            raw,
            events=events,
            tmin=config['tmin'], 
            tmax=config['tmax'], 
            preload=True,
            baseline=None,
            picks='all'
        )
        
        X_train_nn = epochs.copy().pick_types(eeg =True)._data.swapaxes(1, 2)
        y_train_nn = epochs.copy().pick_types(misc=True)._data.swapaxes(1, 2)
        y_train_nn = keras.utils.to_categorical(y_train_nn)

        return X_train_nn, y_train_nn
 