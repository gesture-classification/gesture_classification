from utils.data_reader import DataReader
from utils.inference import MakeInference
from utils.functions import config_reader

if __name__ == "__main__":
    
    path_to_X_test_dataset = 'data/X_test_dataset_2.pkl'
    X_test_dataset = DataReader(path_to_X_test_dataset).data
    
    #path_to_config='config/data_config.json'
    #config = config_reader(path_to_config)
    #path_to_models_weights = config['path_to_models_weights']

    make_inference = MakeInference()
    path_to_models_weights = make_inference.path_to_models_weights

    make_inference.create_prediction(X_test_dataset, path_to_models_weights)