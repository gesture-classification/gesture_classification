# Пересобирать TF без AWS!!!
FROM python:3.10

COPY ./data ./data
COPY ./models_weights ./models_weights
COPY ./utils ./utils
COPY ./req_inference.txt ./req_inference.txt
COPY ./main.py ./main.py
RUN ["mkdir", "./logs_and_figures"]

#RUN pip3 install --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org tensorflow-cpu==2.9.2
RUN pip install --no-cache-dir -r ./req_inference.txt

CMD ["python", "./main.py"]