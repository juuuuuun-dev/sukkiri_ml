FROM python:3.10-buster
ENV PYTHONUNBUFFERED 1
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip \
  && pip3 config list \
  && pip install -r requirements.txt
