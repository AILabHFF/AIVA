# Welcome to Artificial Intelligence Video Analyzer (AIVA)

AIVA is a project of the AI Lab at HFF Munich, which aims not only to simply recognize and classify objects in images, but also to analyze movements and to consider further modalities of videos like sound or text.


# Docker build mit dem tag "aiva" erstellen:
docker build -t aiva .

# Docker Container ausführen (und Volume einbinden):
docker run --gpus all -p 8888:8888 -v /media/disk1/KILabDaten/:/home/jovyan/work --rm  aiva
