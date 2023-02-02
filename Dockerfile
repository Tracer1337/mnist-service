FROM tensorflow/tensorflow

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install flask opencv-python Pillow matplotlib

COPY main.py main.py
COPY mnist_model.h5 mnist_model.h5

EXPOSE 80

CMD flask --app main run --host 0.0.0.0 --port 80
