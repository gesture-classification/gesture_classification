FROM python:3.10.8-alpine

WORKDIR /gesture_classification

COPY ./data .
COPY ./models_weights ./models/models_weights
COPY main.py ./
COPY requirements.txt ./

RUN ["mkdir", "./logs_and_figures"]
#ADD https://github.com/gesture-classification/gesture_classification/blob/main/data/motorica-advanced-gesture-classification.zip \
#./logs_and_figures 

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
