# Motorica Advanced Gesture Classification

## Введение

[В ходе соревнования на Kaggle](https://www.kaggle.com/competitions/motorica-advanced-gesture-classification/leaderboard?) нами была создана математическая модель, предсказывающая жест человека по сокращению мышц предплечья. Модель может использоваться для управления автоматизированным протезом кисти. Задача исследования заключалась в следующем:
* классификация жеста по показаниям показаниям датчиков оптомиографии с поверхности кожи предплечья;
* определение момента начала жеста. 

![Протез](/logs_and_figures/fig_0-1.PNG)

## Структура проекта

1. папка *data* - содержит архив с исходынми данными
2. папка *notebooks* - для ноутбуков проекта
3. папка *logs_and_figures* - для графиков, логов работы модели, сабмиты
4. папка utils - содержит функции
   a. основные в файле *functions.py*
   b. вспомогательные в файле *figures.py*
5. папка inference - 

## Экспериментальные данные

Сбор данных был организован следующим образом: случайным образом было выбрано 3 человека-оператора со здоровыми руками, чтобы обеспечить достоверность показаний. Затем к коже их рук были прикреплены оптомиографические датчики.

![ОМГ](/logs_and_figures/fig_0-2_ru.png)

Оператор циклически выполнял одинаковую последовательность жестов, которую ему демонстрировали. Поскольку оператор тратил некторое время на принятие решения о выполнении показанного жеста, то его жесты выполнялись с некоторой задержкой относительно показанных. Важно, что по условиям эксперимента исходное либо конечное состояние кисти - жест "открытая ладонь". Кроме того, из многообразия жестов были выбраны только характерные положения сгиба и разгиба пальцев для упрощения распознавания. Сигналы манипулятора были записаны файлы y_train, а сигналы с датчиков оптомиографии были записаны в файлы X_train и X_test. 

![Raw data](/logs_and_figures/fig_2-1.png)

## Подготовка и анализ данных

Предварительная обработка данных заключалась в:
* поиске начала жеста математическими методами
* удалении "нечитаемых" участков (класс жеста "-1") с помощью нормализации данных и производных сигналов 
  
# тут добавить нужно текст и картинки


Поскольку качество модели существенно не изменяется после предобработки, то было решено отказаться от неё ([*см. ноутбук 1_EDA_sprint_3.ipynb*](https://github.com/gesture-classification/gesture_classification/blob/main/notebooks/1_EDA_sprint_3.ipynb)). 



Данные загружаются из архива и преобразуются с помощью библиотеки [*mne*](https://mne.tools/stable/index.html) для последующей подачи данных на обучение. Далее последовательно обучаются две модели: 

- SimpleRNN (первая модель на базе слоя SimpleRNN библиотеки [*Keras*](https://keras.io/)); 

- LSTM (вторая модель, в ее основе лежат несколько слоев LSTM библиотеки Keras и дополнительный Dense-слой). 

Важно отметить, что тренировочные и тестовые данные имеют разделение на 3 ряда данных и по каждому набору происходит параллельное обучение группы моделей, имеющих одинаковую структуру и набор параметров. 

Основная задача работы первой модели - определить фактический момент изменения жеста (появление "ступеньки") по данным X_train для последующего обучения более сложной модели. Использование упрощенной модели SimpleRNN совместно с использованием loss="mean_squared_error" и функцией активации 'sigmoid' (activation='sigmoid') в выходном слое при сборке модели позволяет сделать предсказание "ступеньки" при решении задачи классификации жестов по данным датчиков (X_train). Модель учитывает классы из y_train, а время выполнения движения определяется из предикта по X_train как момент изменения класса (жеста). 

Необходимость первого этапа обусловлена спецификой подготовки данных для обучения, когда человек ("пилот") с зафиксированным на запястье набором датчиков повторяет жесты следуя командам манипулятора. Таким образом, изначально y_train представляет собой момент подачи манипулятором команды на изменение жеста, а данные X_train - фактическое выполнение жеста - запаздывают на некоторое время относительно исходного y_train. 

Для того, чтобы компенсировать ошибки предсказания первой модели, обучение SimpleRNN по каждому "пилоту" проводится несколько раз с разными параметрами validation_split и затем результаты предсказания каждой модели усредняются по каждому пилоту. 

Обучение второй модели производится на оригинальных данных X_train и корректированных данных y_train_ch (предсказание обученной модели SimpleRNN на X_train). Далее обученная модель LSTM используется для предсказания тестовых данных. 

При работе с моделями для управления обучением (выбор лучшей модели, изменение learning_rate, остановка обучения при выходе на плато) используется набор функций *callbacks* библиотеки Keras.



# выводы




## 
- как запустить проект + код с инференсом 



  
__________________________________________





**3)** В ноутбуке [*3_embeddings.ipynb*](https://github.com/Alex1iv/Motorica_3/blob/main/3_embeddings.ipynb) реализовано предсказание тестовых данных на модели, обученной для каждого пилота. Модели выложены в папке [*lstm_model*](https://github.com/Alex1iv/Motorica_3/tree/main/lstm_model).

**4)** Файл [*4_boxplots_clear_gests_sens_gest.ipynb*](https://github.com/Alex1iv/Motorica_3/blob/main/4_boxplots_clear_gests_sens_gest.ipynb) с построением боксплотов "Статистика изменения характерных уровней датчиков в течение снятия показаний в разрезе жестов для выбранного пилота. Очищенные данные" и папка [*boxplots*](https://github.com/Alex1iv/Motorica_3/tree/main/boxplots) с боксплотами, построенными для всех пилотов. В том числе [*boxplots_sens_gest_pylot2_with_beaten.png*](https://github.com/Alex1iv/Motorica_3/blob/main/boxplots/boxplots_sens_gest_pylot2_with_beaten.png), построенный по данным 2-го пилота, еще не очищенным от битых участков. Обсуждение наблюдений - в общем файле с разведочным анализом данных [*1_EDA_sprint_3.ipynb*](https://github.com/Alex1iv/Motorica_3/blob/main/1_EDA_sprint_3.ipynb).

**5)** [*5_rnn_baseline.ipynb*](https://github.com/Alex1iv/Motorica_3/blob/main/5_rnn_baseline.ipynb) - ноутбук, предоставленный организаторами соревнования в качестве **baseline**.   

**6)** Папка [*data*](https://github.com/Alex1iv/Motorica_3/tree/main/data) содержит [архив](https://github.com/Alex1iv/Motorica_3/tree/main/data) с исходными данными:

- *X_train_1.npy*, *X_train_2.npy*, *X_train_3.npy*: файлы с тренировочными данными ("фичи", показания датчиков по каждому "пилоту");

- *y_train_1.npy*, *y_train_2.npy*, *y_train_3.npy*: файлы с тренировочными "таргетами" (от манипулятора);

- *X_test_dataset_1.pkl*, *X_test_dataset_2.pkl*, *X_test_dataset_3.pkl*: файлы тестовых данных ("фичи", показания датчиков по каждому "пилоту") для предсказания и сабмита;

- *sample_submission.csv*: файл примера загрузки предсказанных данных на Kaggle.

**7)** Файлы с агрегированными предиктами обученных моделей SRNN+LSTM на тестовых данных, показавшие максимальный score на Leaderboard при сабмите 

*y_test_submit_rnn_LSTM(0.69641).csv*,

*y_test_submit_rnn_LSTM(0.68976).csv*,

*y_test_submit_rnn_LSTM(0.6781).csv*
