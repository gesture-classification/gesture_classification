# Motorica Advanced Gesture Classification

## Оглавление

* [Введение](README.md#Введение)
* [Структура проекта](README.md#Структура-проекта)
* [Экспериментальные данные](README.md#Экспериментальные-данные)
* [Установка](README.md#Установка)
* [Подготовка и анализ данных](README.md#Подготовка-и-анализ-данных)
* [Структура моделей машинного обучения](README.md#Структура-моделей-машинного-обучения)
* [Выводы](README.md#выводы)
* [Лог эксперимента](README.md#лог-эксперимента)




## Введение

[В ходе соревнования на Kaggle](https://www.kaggle.com/competitions/motorica-advanced-gesture-classification/leaderboard) нами была создана математическая модель, предсказывающая жест человека по сокращению мышц предплечья. Модель может использоваться для управления автоматизированным протезом кисти. Задача исследования заключалась в следующем:
* классификация жеста по показаниям датчиков оптомиографии с поверхности кожи предплечья;
* определение момента начала жеста.

<p align="center"> 
   <img src="/logs_and_figures/fig_0-1.PNG" height=300>
</p>


## Структура проекта

* папка *data* - содержит архив с исходными данными
* папка *logs_and_figures* - содержит графики, логи работы модели, сабмиты
* папка *models* - модели проекта и их коэффициенты 
* папка *notebooks* - ноутбуков проекта
* папка utils - содержит функции
    * основные в файле *functions.py*
    * вспомогательные в файле *figures.py*

:arrow_up:[Оглавление](README.md#оглавление)

## Экспериментальные данные

Сбор данных был организован следующим образом: случайным образом было выбрано 3 человека-оператора со здоровыми руками, чтобы обеспечить достоверность показаний. Затем к коже их рук были прикреплены оптомиографические датчики.

<p align="center"> 
   <img src="/logs_and_figures/fig_0-2_ru.png" height=250>
</p>

Оператор циклически выполнял одинаковую последовательность жестов, которую ему демонстрировали. Поскольку оператор тратил некоторое время на принятие решения о выполнении показанного жеста, то его жесты выполнялись с некоторой задержкой относительно показанных. Важно, что по условиям эксперимента исходное либо конечное состояние кисти - жест "открытая ладонь". Кроме того, из многообразия жестов были выбраны только характерные положения сгиба и разгиба пальцев для упрощения распознавания. Сигналы манипулятора были записаны файлы y_train, а сигналы с датчиков оптомиографии были записаны в файлы X_train и X_test. 

![Raw data](/logs_and_figures/fig_2-1.png)

:arrow_up:[Оглавление](README.md#оглавление)

## Установка
1. Скопируйте репозиторий, введя в терминале следующие команды:
```
# Clone repository and install requirements

git clone https://github.com/gesture-classification/gesture_classification
pip install -r -q requirements.txt
``` 
2. Переместите архив с данными для обучения (X_train) и валидации (X_test)  в папку *data*. 
3. Выполните в терминале команду ```python main.py```

## Подготовка и анализ данных

Тренировочные и тестовые данные каждого пилота загружаются из архива. Они преобразуются с помощью библиотеки [*mne*](https://mne.tools/stable/index.html).

По результатам анализа сделаны следующие выводы:
* удаление "нечитаемых" участков (класс жеста "-1") позволяет легко избавиться от "битых" данных
* можно выделить характерные уровни сигнала датчиков при каждом жесте ( см. файл [*3_boxplots_clear_gests_sens_gest.ipynb*](https://github.com/gesture-classification/gesture_classification/blob/main/notebooks/3_boxplots_clear_gests_sens_gest.ipynb)
* датчики можно разделить по величине на активные и пассивные;
  
![активные и пассивные датчики](/logs_and_figures/fig_1-3.png)

* нормализация сигналов датчиков и взятии производной от функции не позволил однозначно определить временной интервал начала жеста.
<p align="center">   <img src="/logs_and_figures/fig_1-5.png"> </p>

Сравнительный анализ предсказаний моделей до и после предобработки показал, что все методы, кроме первого, ухудшают качество предсказания. Поэтому было решено отказаться от второго этапа и обучать модель на "сырых" данных. ([*см. ноутбук 1_EDA_sprint_3.ipynb*](https://github.com/gesture-classification/gesture_classification/blob/main/notebooks/1_EDA_sprint_3.ipynb)). 

:arrow_up:[Оглавление](README.md#оглавление)

## Структура моделей машинного обучения

Наборы тренировочных и тестовых данных каждого пилота поступают на параллельное обучение двух моделей, имеющих одинаковую структуру и набор параметров: 
- SimpleRNN, задача которой предсказать фактический момент изменения жеста (появление "ступеньки");
- LSTM, которая предсказывает класс жеста с учётом времени начала жеста, определённого первой моделью. 

![поиск временного интервала начала жеста](/logs_and_figures/fig_2-2.png)

Модель SimpleRNN из библиотеки [*Keras*](https://keras.io/) имеет простую структуру из одного слоя. Чтобы компенсировать ошибки предсказания, обучение модели для каждого "пилота" проводится несколько раз с разными параметрами validation_split и затем результаты предсказания каждой модели усредняются по каждому пилоту. Ошибка предсказания определяется по среднеквадратическому отклонению.

:arrow_up:[Оглавление](README.md#оглавление)


### Добавить график обучения первой модели
![график обучения первой модели](/logs_and_figures/fig_.png)

Вторая модель состоит из нескольких слоев LSTM библиотеки Keras с дополнительным Dense-слоем. Структура слоёв модели подбиралась эмпирически по оценке f1-score. Обучение модели производится на оригинальных данных X_train и корректированных данных y_train_ch. Затем обученная модель LSTM используется для предсказания тестовых данных.

По наклону кривой обучения на рис.2-3 видно, что модель переобучается. Неоптимально выбранные параметры обучения повышают время обучения и снижают точность предсказании тестовых данных. Однако, полученный результат f1-score составляет 0.69632.

![график обучения второй модели](/logs_and_figures/fig_2-3.png)

:arrow_up:[Оглавление](README.md#оглавление)


## Выводы
Разработанная модель предсказания жеста по мускульному сигналу предплечья имеет удовлетворительную точность $\approx 70$ %. 

:arrow_up:[Оглавление](README.md#оглавление)


## Лог эксперимента

Параметры и ход обучения модели, а также графики были залоггированы с помощью [библиотеки Comet_ml](https://www.comet.com/alex1iv/gesture-classification/view/new/panels).

