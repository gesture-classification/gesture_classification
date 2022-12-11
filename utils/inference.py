import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import csv
import numpy as np
from tensorflow import keras
from utils.functions import f1
#from utils.data_loader import DataLoader
from utils.data_reader import DataReader
from utils.reader_config import read_config


class MakeInference():

    def __init__(self, path_to_X_test_dataset, 
                 path_to_config='config\data_config.json'):
        super(MakeInference, self).__init__()
        self.config = read_config(path_to_config)
        #self.data_loaded = DataLoader('config\data_config.json')
        
        #self.X_train_nn = self.data_loaded.X_train_nn
        self.X_test_dataset = DataReader(path_to_X_test_dataset).data
        self.path_to_models_weights = self.config['path_to_models_weights']
        #self.y_pred = self.create_prediction(self.X_train_nn, self.X_test_dataset, 
        #                                     self.path_to_models_weights)
        self.create_prediction(self.X_test_dataset, #self.X_train_nn, 
                               self.path_to_models_weights)

    def create_prediction(self, X_test_dataset, path_to_models_weights): #X_train_nn, 

        #m_lstm = keras.models.load_model(path_to_models_weights, custom_objects={"f1": f1})

        m_lstm = keras.models.load_model(path_to_models_weights, compile=False)
        m_lstm.compile(loss="mean_squared_error", metrics=[f1], optimizer=keras.optimizers.Adam())

        #m_lstm.predict(X_train_nn, verbose=0)
        
        y_pred_test_lstm = []

        for i in range(len(X_test_dataset)):
            X_test_i = np.expand_dims(X_test_dataset[i], axis=0).swapaxes(1, 2).astype(np.float64)
            y_pred_test_lstm += [m_lstm.predict(X_test_i, verbose=0)]
        
        y_pred_test_lstm = [arr.argmax(axis=-1) for arr in y_pred_test_lstm]
        
        with open("predictions.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(y_pred_test_lstm)
        print('X_test_dataset predicted!')
        #return y_pred_test_lstm