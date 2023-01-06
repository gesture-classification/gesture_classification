from utils.data_reader import DataReader
from utils.inference import MakeInference
from utils.functions import config_reader


if __name__ == "__main__":
    '''
    Введите номер пилота, по которому загрузить данные X_test
    для получения predict с помощью уже обученной на данных этого пилота модели
    '''
    id_pilot = 3
    path_to_config = 'config/data_config.json'
    config = config_reader(path_to_config)
        
    path_to_X_test_dataset = config.PATH[3:] + config.mounts[str(id_pilot)].path_X_test_dataset
    X_test_dataset = DataReader(path_to_X_test_dataset).data

    MakeInference.create_prediction(X_test_dataset, config, id_pilot)
