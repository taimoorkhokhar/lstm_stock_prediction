FROM python:3.7
RUN apt-get -y update
ENV PYTHONUNBUFFERED=1
WORKDIR /code
ADD requirements.txt /code/
RUN pip install -r requirements.txt
COPY . ./
CMD [ "python3.7", "run_lstm.py"]
