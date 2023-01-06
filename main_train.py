from utils import one_learning
from utils.functions import config_reader


if __name__ == "__main__":
    '''
    Введите номер пилота, по которому загрузить данные X_train и y_train
    для обучения и сохранения обученной модели
    '''
    id_pilot = 2
    path_to_config = 'config/data_config.json'
    config = config_reader(path_to_config)
    
    learning_pilot = one_learning.OneLearning(config)
    learning_pilot.save_lstm_model(id_pilot=id_pilot)
    