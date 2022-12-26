from utils.data_reader import DataReader
#from utils.functions import config_reader

import numpy as np
import mne
import os
import pandas as pd
from tensorflow import keras


class DataLoader():
    """Программа загрузки X_train и y_train для подачи в модель.
    Аргументы:
    -------------
    config - 
    X_train - 
    y_train - 
    path_X_train - 
    path_y_train - 
    id_pilot - 
    """    
    def __init__(self, id_pilot, config):
        super(DataLoader, self).__init__()
        self.config = config
        self.X_train = DataReader(
            self.config['PATH'] + self.config['mounts'][str(id_pilot)]['path_X_train']).data
        self.y_train = DataReader(
            self.config['PATH'] + self.config['mounts'][str(id_pilot)]['path_y_train']).data
        #self.X_train = DataReader(self.config['path_to_x_trn']).data
        #self.y_train = DataReader(self.config['path_to_y_trn']).data
        self.train_nn = self.create_train_nn(self.X_train, self.y_train, self.config)
        self.X_train_nn = self.train_nn[0]
        self.y_train_nn = self.train_nn[1]
        

    def create_train_nn(self, X_train, y_train, config):
    
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

        return (X_train_nn, y_train_nn)
 