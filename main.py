from utils.functions import choice_train_inference


if __name__ == "__main__":
    print('Программа позволяет сделать predict для тестовых данных по номеру пилота\n'
          'на основании предобученных в рамках проекта моделей.\n'
          'Или заново обучить модель на данных X_train и y_train по номеру пилота.\n')
    choice_train_inference()
