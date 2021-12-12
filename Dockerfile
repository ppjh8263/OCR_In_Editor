FROM python:3.7.11-slim-buster

COPY . /workdir
WORKDIR . /workdir
ENV PYTHONPATH=/workdir
ENV PYTHONUNBUFFERED=1

RUN pip install pip==21.0.1 && \
    pip install -r requirements.txt

CMD ["nohup", "python", "models/api.py", ">>", "models/server/logs/docker.txt"]