# pull the official base image
FROM python:3.8

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

# install dependencies
RUN pip install --upgrade pip 
COPY ./laneDetection/requirements.txt /usr/src/app
RUN pip install -r requirements.txt




# copy project
COPY ./laneDetection /usr/src/app

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
