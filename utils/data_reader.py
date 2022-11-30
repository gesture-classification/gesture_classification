import numpy as np
from zipfile import ZipFile
import mne
import pickle
import pandas as pd


class DataReader():
    
    def __init__(self, path_to_zip):
        self.path_to_zip = path_to_zip


    def read_data(self):
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

        SFREQ = 1000.0 / 33

        for mount_name, mount in mounts.items():
            mount['X_train'] = np.load(self.path_to_zip)[mount['path_X_train']]
            mount['y_train'] = np.load(self.path_to_zip)[mount['path_y_train']]
            with ZipFile(self.path_to_zip) as myzip:
                with myzip.open(mount['path_X_test_dataset']) as myfile:
                    mount['X_test_dataset'] = pickle.load(myfile)
            
            X_train = mount['X_train'] 
            y_train = mount['y_train']
            
            raw = mne.io.RawArray(
                data=X_train.T,
                info=mne.create_info(
                    ch_names=list(np.arange(X_train.shape[1]).astype(str)),
                    sfreq=SFREQ,
                    ch_types='eeg'
                )
            )
            raw_y = mne.io.RawArray(
                data=y_train.reshape(1,-1),
                info=mne.create_info(
                    ch_names=['y'],
                    sfreq=SFREQ,
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
                tmin=-1, 
                tmax=1*2.5, 
                preload=True,
                baseline=None,
                picks='all'
            )
            
            X_train_nn = epochs.copy().pick_types(eeg =True)._data.swapaxes(1, 2)
            mount['X_train_nn'] = X_train_nn

        return mounts

    
    def read_y_test(self):
        # Чтение sample_submission.csv из архива
        with ZipFile(self.path_to_zip) as myzip:
            y_test = pd.read_csv(myzip.open('sample_submission.csv'))

        y_test[['subject_id', 'sample', 'timestep']] = (
            y_test['subject_id-sample-timestep']
            .str.split('-', 2, expand=True)
            .astype(int)
        )
        return y_test    