FROM python:3.8


ADD . /

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install numpy 
RUN pip install opencv-contrib-python --upgrade


CMD ["python","./laneDetection.py"]