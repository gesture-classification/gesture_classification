import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
from tensorflow import keras
from utils.functions import f1



class MakeInference():

    def __init__(self, mounts, y_test, path_to_models_weights):
        self.mounts = mounts
        self.y_test = y_test
        self.path_to_models_weights = path_to_models_weights


    def add_prediction(self, mount, mount_name, y_test):

        X_train_nn = mount['X_train_nn']
        X_test_dataset = mount['X_test_dataset']
        m_lstm = keras.models.load_model(os.path.join(self.path_to_models_weights, 
                                                      'model_lstm_' + str(mount_name)), 
                                         custom_objects={"f1": f1})
        m_lstm.predict(mount['X_train_nn'], verbose=0)
        
        y_pred_test_lstm = []

        for i in range(len(X_test_dataset)):
            X_test_i = np.expand_dims(X_test_dataset[i], axis=0).swapaxes(1, 2).astype(np.float64)
            y_pred_test_lstm += [m_lstm.predict(X_test_i, verbose=0)]
        
        y_pred_test_lstm = [arr.argmax(axis=-1) for arr in y_pred_test_lstm]
        print(len(y_pred_test_lstm))
        assert len(y_pred_test_lstm) == y_test.query("subject_id == @mount_name")['sample'].nunique()
        
        mount['y_pred_test_lstm'] = y_pred_test_lstm
        return mount


    def make_inference(self):
        
        for mount_name, mount in self.mounts.items():
            mount = self.add_prediction(mount, mount_name, self.y_test)

        y_pred_test_res = []
        
        for mount_name, mount in self.mounts.items():
            y_pred_test_res.extend(mount['y_pred_test_lstm'])
        y_pred_test_res = np.concatenate(y_pred_test_res, axis=-1)[0]
        
        assert y_pred_test_res.shape[0] == self.y_test.shape[0]
        
        y_test_submit = self.y_test[['subject_id-sample-timestep', 'class']]
        y_test_submit['class'] = y_pred_test_res
        y_test_submit.to_csv('logs_and_figures/y_test_submit_from_inference.csv', index=False)
        return y_test_submit