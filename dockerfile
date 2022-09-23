# Use the image base-notebook to build our image on top of it
FROM python:3.8-slim-buster

# Set Workdir
#WORKDIR /app

# Change to root user
USER root

# Set Environment to ignore sudo warning for pip
ENV PIP_ROOT_USER_ACTION=ignore

# Copy data from current directory into the docker image
COPY . .

# Install package requirements
RUN pip3 install --upgrade pip
#RUN apt-get update && apt-get install libgl1 -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install -r requirements.txt

# Set
#CMD [ "python3", "-m" , "main.py", "--host=0.0.0.0"]
CMD python3 main.py