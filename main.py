from utils.data_reader import DataReader
from utils.inference import MakeInference

# библиотека взаимодействия с интерпретатором
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    print('Working ...')
    path_to_zip = 'data\motorica-advanced-gesture-classification.zip'
    data_reader = DataReader(path_to_zip)
    mounts = data_reader.read_data()
    y_test = data_reader.read_y_test()
    print('Data loaded!', end='\n\n')
    
    print('Predicting test data ...')
    path_to_models_weights = 'gesture_classifiaction/models/models_weights'  # было models_weights
    y_test_submit = MakeInference(mounts, y_test, path_to_models_weights).make_inference()
    print(f'y_test_submit_from_inference.csv created in folder: logs_and_figures')