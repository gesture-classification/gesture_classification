# Пересобирать TF без AWS!!!
FROM python:3.10

WORKDIR /usr/src/app

COPY ./config ./config
COPY ./data ./data
COPY ./models ./models
COPY ./utils ./utils
COPY ./requirements.txt ./requirements.txt
COPY ./main_inference.py ./main_inference.py

#RUN pip3 install --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org tensorflow-cpu==2.9.2
RUN pip install --no-cache-dir -r ./requirements.txt

CMD ["python", "./main_inference.py"]

LABEL container.name='gesture' 

