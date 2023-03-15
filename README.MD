# MAD Midnight (Meteor Anomaly Detector) WebUI

The MAD Midnight WebUI is a state-of-the-art web application that uses two powerful machine learning models to provide real-time detection and classification of anomalies. The app can accurately identify objects as either meteors or other events, making it a valuable tool for researchers, scientists, and anyone else who requires precise anomaly detection.

Notably, the app is highly accessible and can be used remotely from anywhere in the world with an SSH connection to the server. This means that users can easily perform analyses and access results without being physically present at the server location.

![](data/WebUIAnimation.gif)

# Installation
- Install [Python3](https://www.python.org/downloads/release/python-3100/) (Version 3.7 - 3.10)
### On Linux:
```
chmod +x webui.sh
./webui.sh
```

### On Windows:
```
double click the webui.bat 
```
then open the given link in yout Internet Browser



# Docker
Alternatively, you can use Docker to build and run MAD Midnight WebUI. Follow these steps:

## Build the Docker image with the tag "madmidnight" by running:
docker build -t madmidnight .

## Run the Docker container with the madmidnight image, including a volume, by running:
docker run --gpus all -p 8888:8888 -v /media/disk1/KILabDaten/:/home/jovyan/work --rm  madmidnight
