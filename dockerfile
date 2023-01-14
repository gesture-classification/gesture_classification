FROM python:3.10

WORKDIR /gesture_classification

COPY ./config ./config
COPY ./data ./data
COPY ./models ./models
COPY ./utils ./utils
COPY ./requirements.txt ./requirements.txt
COPY ./main.py ./main.py

RUN pip install --no-cache-dir -r ./requirements.txt

CMD ["python", "./main.py"]