# Advanced Gesture Classification

## Contents

* [Introdcution](README.md#Contents)
* [Project structure](README.md#Project-structure)
* [Data and methods](README.md#Data-and-methods)
* [Installation](README.md#Installation)
* [Inference](README.md#Inference)
* [Experimental data analysis](README.md#Experimental-data-analysis)
* [Machine learning models structure](README.md#Machine-learning-models-structure)
* [Experiment logging](README.md#Experiment-logging)
* [Conclusion](README.md#Conclusion)


## Introdcution

When a human loses an organ or limb loss, his functions decrease dramatically. Luckily technological progress in biomedicine provided us with prostheses like artificial kidneys, mechanical ventilation, artificial heart valves, bionic hand, etc., so lost body functions could be partially recovered. The compensated-to-original functions ratio depends on multiple factors such as the complexity of the lost organ, the number of original functions and their variability, and so on. Due to the large number of finger gestures, hand prosthetics is considered a very complex assignment. 

In 2020 in Russia there were about 20,000 people with upper limbs amputations, half of them missing forearm or wrist([Grinin V. M., Shestemirova E. I.](http://dx.doi.org/10.32687/0869-866X-2020-28-3-380-384). Notably, only some 500 people were able to get operated prostheses from the national prosthetics program [Rosstat](https://rosstat.gov.ru/storage/mediabank/ST90u1EJ/2-30.doc). Thus, due to the lack of prostheses supply and because of its direct impact on human life quality functional hands prosthetics is also of high importance.

To provide better functional wrist prostheses the company [Motorica](https://motorica.org/) designed [a Kaggle competition](https://www.kaggle.com/competitions/motorica-advanced-gesture-classification) with the task of predicting wrist gestures from muscle contractions.

During the competition it was designed a predictive mathematical model, conducting the following tasks:
* gesture change period identification.
* gesture classification by muscle signals from optomyographical $^1$ sensors.


<p align="center"> 
<img src="/logs_and_figures/fig_0-1.PNG" width="450" height="350"> <br>image from https://motorica.org/ 
</p>


## Project structure

<details>
  <summary>display project structure </summary>

```Python
gesture_classification
├── .git
├── .gitignore
├── data             # data archive
│   └── motorica-advanced-gesture-classification.zip
├── dockerfile
├── logs_and_figures # charts, logs, and submits
│   ├── fig_0-1.PNG
...
│   ├── fig_2-5.png
│   ├── y_test_submit_rnn_LSTM(0.69641).csv
│   └── y_test_submit_rnn_LSTM.csv
├── main.py
├── models           # models and weights 
│   ├── best_model_rnn_1.hdf5
│   ├── best_model_rnn_2.hdf5
│   ├── best_model_rnn_3.hdf5
│   ├── lstm.py
│   ├── model.py
│   ├── model_lstm_1
│   ├── model_lstm_2
│   ├── model_lstm_3
│   ├── srnn.py
│   ├── temp_best_model
│   └── weights
├── notebooks        # notebooks
│   ├── .cometml-runs
│   ├── 1_EDA_sprint_3.ipynb
│   ├── 2_model_SRNN_LSTM.ipynb
│   └── 3_boxplots_clear_gests_sens_gest.ipynb
├── README.md
├── requirements.txt 
└── utils            # functions, variables, and data loaders
    ├── credentials.json
    ├── data_reader.py
    ├── figures.py
    ├── functions.py
    ├── inference.py
    ├── __ init __.py
    └── __pycache__
```
</details>


## Data and methods
Data acquisition was conducted as follows:  there were chosen 3  people (operators) with healthy hands and on which were attached 50 optomyographical $^1$ sensors. Then each operator was bending and unclenching his/her fingers, performing a given sequence of gestures.

$^1$ Optomyography (OMG) is a method of monitoring muscle activity with optical sensors. OMG infrared light source emits impulses toward the muscle under the skin. If the muscle is neutral, the light will be almost completely reflected; in case it is stretched or compressed - partially diffused. So the amount of diffused light is measured by a light detector(Fig.1-2).

<p align="center"> 
<img src="/logs_and_figures/fig_0-2_en.png" width="500" height="200"> <br>Fig.1-1 - Muscle tension detection using the optomyography method.</p>

The experiment details were as follows: since every operator takes some time between observing a gesture and performing a gesture, it can be seen a gap between the two line graphs below. On the second, each gesture was started or ended with the "open palm" gesture. On the third, from the whole set of gestures only distinctive ones were: either with clenching or unclenching fingers to simplify recognition. As result, experimental data was split into 2 parts, and the sequence of original gestures was saved into arrays X_train, X_test, and y_train respectively. An example of the acquired data is represented in the figure below.

![Experimental dara](/logs_and_figures/fig_2-1.png)
<p align="center">Fig.1-2 - Experimental data of pilot #3.</p>

:arrow_up:[ to contents](README.md#Contents)

## Installation
Type in the console:
```Python
# 1. Clone repository and install requirements.

git clone https://github.com/gesture-classification/gesture_classification
pip install -r -q requirements.txt
 
# 2. Make sure the experimental data X_train and X_test files are in the data folder. 
# 3. Create the model.

python main.py
```

## Inference
<details>
  <summary> Display how to get an inference </summary>
<br>

The term **inference** means gesture prediction using a fully trained machine learning model from user data. The inference is performed using a class *MakeInference*, which takes the path to test data (*path_to_X_test_dataset*) as an argument and saves the prediction into the file "predictions.csv" in the project root directory. The methods of the class are as follows:
- loading of variables from *data_config.json*; 
- loading of train and test data using *DataReader*; 
- loading of a pre-trained model from the *models* folder;
- learning on the given data;
- saving the output file *predictions.csv*.


### Get an inference from

1. Put your initial data in format `*.npy` into the *data* folder. Each time series of user data must contain 50 features from OMG sensors with any time duration.
2. Assign paths to the following files to variables in the config *data_config.json*, namely:
   * `path_to_x_trn` - train data *X_train*
   * `path_to_y_trn` - test data *X_test*
   * `path_to_x_tst` - target variable *y_train*
   * `path_to_models_weights` - model
   

Example:
```Python
{
  "path_to_x_trn" : "data/X_train_2.npy",
  "path_to_y_trn" : "data/y_train_2.npy",
  "path_to_x_tst" : "data/X_test_dataset_2.pkl",
  "path_to_models_weights" : "models/model_lstm_2",
}
```
</details>

## Experimental data analysis

Original experimental data of all pilots was loaded from the archive and processed using the [*mne*](https://mne.tools/stable/index.html) library as follows:
* writing an array of raw dataset for each pilot
* deleting some cropped and unreadable data of class "-1".

In result of the EDA ([see notebook 1_EDA_sprint_3.ipynb*](https://github.com/gesture-classification/gesture_classification/blob/main/notebooks/1_EDA_sprint_3.ipynb)), it was inferred:
* each gesture can be identified by distinctive signal levels (see [*3_boxplots_clear_gests_sens_gest.ipynb*](https://github.com/gesture-classification/gesture_classification/blob/main/notebooks/3_boxplots_clear_gests_sens_gest.ipynb)
* sensors can be categorized by the level of the signal into active and passive (see Fig.1-3);

![Active and passive sensors](/logs_and_figures/fig_1-3.png)

* signals processing methods namely normalization and derivative were useless for the identification of gesture change as can be seen in Fig.1-5).  
  
![Signals normalization](/logs_and_figures/fig_1-5.png)

A comparative analysis of the models, trained with EDA and without showed that chosen signal processing methods worsened prediction. Thus, it was chosen to avoid signal processing and use raw data for model training . 

:arrow_up:[ to contents](README.md#Contents)


## Machine learning models structure

Every set of train and test data learns 2 models, having a similar structure as shown in Fig.2-1 and parameters as follows:
- SimpleRNN, used for labeling the data by predicting the period of gesture change (it is expressed as a "step" on the graph);
- LSTM, which is learning from OMG signals and the labeling data. 
  
![Model structure](/logs_and_figures/fig_2-2.png)
<p align="center">  Fig.2-1 - Machine learning model structure. </p>

**SimpleRNN model** (from the [*Keras*](https://keras.io/) library) has a simple structure and contains 1 layer. To compensate the prediction error, the data was split several times into parts with different ratios (validation_split) and every model predicts pilots' gests. As a result, the mean of all predictions was taken. The prediction quality is estimated using the mean squared error metrics. The learning progress can be observed in Fig.2-2.


![Learning progress of the model SimpleRNN](/logs_and_figures/fig_2-3.png)
<p align="center">  Fig.2-2 - SimpleRNN model learning progress. </p>


As can be seen from Fig.2-2, the predictive quality does not change much. On the other hand, the model predicts the gesture change period successfully.

**LSTM model** consists of several LSTM-layers with an additional dense layer. The layer structure was empirically defined from the best f1-score. Model training was carried on the dataset *X_train* and labeled data *y_train_ch*. Then trained LSTM model was used to predict the test data. The learning progress can be observed in Fig.2-3.

![LSTM model learning progress](/logs_and_figures/fig_2-5.png)

<p align="center">  Fig.2-3 - LSTM model learning progress. </p>

The u-shape pattern of the learning curve in Fig.2-3 points on the model overfitting. Non-optimally chosen parameters are leading to an increase in the learning time and growth of prediction errors. Meanwhile, it reached fair model quality predicting correctly some of 2/3 of datapoints  (f1-score is 0.69632). The result can be observed in the graph provided below.

![Gesture prediction](/logs_and_figures/fig_2-6.png)

<p align="center">  Fig.2-4 - Original signals and gesture classification </p>

:arrow_up:[ to contents](README.md#Contents)

## Experiment logging

The learning progress as well as other model parameters, figures, and variables were logged using the Comet_ml library. These outcomes can be observed on the [Comet_ml page](https://www.comet.com/alex1iv/gesture-classification/view/new/panels).


## Conclusion
As a result of the study, it was created the data processing algorithm and the machine learning model, which is predicting wrist gestures by muscle signals. The model can be used to create better hand prostheses.

:arrow_up:[ to contents](README.md#Contents)