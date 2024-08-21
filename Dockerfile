FROM continuumio/miniconda3

COPY requirements.txt .

RUN apt-get update \ 
    && apt-get -y install g++ gcc libsm6 libxext6 cron pciutils libgl1-mesa-glx

RUN apt -y install default-jre
RUN apt-get update && apt-get install -y openjdk-17-jdk && apt-get clean;
RUN pip install -r requirements.txt

COPY data data
COPY docs docs
# COPY evaluate evaluate
# COPY models models
COPY src src
COPY run.py run.py
COPY model_downloader.py model_downloader.py

ENTRYPOINT ["python", "model_downloader.py"]

EXPOSE 8000 8501

ENTRYPOINT ["python", "run.py"]


