from utils import one_learning
from utils.data_reader import DataReader
from utils.inference import MakeInference
from utils.functions import config_reader


if __name__ == "__main__":
    while True:
        print('Программа позволяет сделать predict для тестовых данных 1, 2 или 3 пилота')
        print('на основании предобученных в рамках проекта моделей.')
        print('Или заново обучить модель на данных X_train и y_train для 1, 2 или 3 пилота.')
        print('Выберите нужную опцию:')
        print('     1 - сделать предсказание для тестовых данных на основании предобученной модели,')
        print('     2 - обучить новую модель для произвольного пилота,')
        print('     3 - выйти.')

        choice = str(input('Введите 1, 2 или 3: '))
        if choice == '1':
            print('Введите номер пилота, по которому загрузить данные X_test')
            print('для получения predict с помощью уже обученной на данных этого пилота модели')
            id_pilot = int(str(input('Введите 1, 2 или 3: ')))

            if id_pilot not in (1, 2, 3):
                id_pilot = int(str(input('Введите номер пилота 1, 2 или 3,\nдругой выбор - выйти: ')))
                if id_pilot not in (1, 2, 3):
                    break

            print('Подождите, идет расчет...')
            path_to_config = 'config/data_config.json'
            config = config_reader(path_to_config)

            path_to_X_test_dataset = config.PATH[3:] + config.mounts[str(id_pilot)].path_X_test_dataset
            X_test_dataset = DataReader(path_to_X_test_dataset).data

            MakeInference.create_prediction(X_test_dataset, config, id_pilot)

        elif choice == '2':
            print('Введите номер пилота, по которому загрузить данные X_train и y_train')
            print('для обучения и сохранения обученной модели')

            id_pilot = int(str(input('Введите 1, 2 или 3: ')))

            if id_pilot not in (1, 2, 3):
                id_pilot = int(str(input('Введите номер пилота 1, 2 или 3,\nдругой выбор - выйти: ')))
                if id_pilot not in (1, 2, 3):
                    break

            print('Подождите, идет расчет...')
            path_to_config = 'config/data_config.json'
            config = config_reader(path_to_config)

            learning_pilot = one_learning.OneLearning(config)
            learning_pilot.save_lstm_model(id_pilot=id_pilot)

        elif choice == '3':
            break
        else:
            continue
