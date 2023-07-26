FROM python:3.8
COPY . /app
WORKDIR /app
RUN conda create -p venv python==3.8 &&\
	pip install -r Requirements.txt &&\
	pyhton test_main.py
	