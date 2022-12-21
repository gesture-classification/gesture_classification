import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import numpy as np
from tensorflow import keras
from utils.functions import f1
from utils.functions import config_reader


class MakeInference():

    def __init__(self, 
                 path_to_config='config/data_config.json'
                ):
        super(MakeInference, self).__init__()
        self.config = config_reader(path_to_config)
        self.path_to_models_weights = self.config['path_to_models_weights']

    def create_prediction(self, X_test_dataset, path_to_models_weights):

        m_lstm = keras.models.load_model(path_to_models_weights, compile=False)
        m_lstm.compile(loss="mean_squared_error", metrics=[f1], optimizer=keras.optimizers.Adam())

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