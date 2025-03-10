FROM python:3.9-slim

COPY requirements.* ./
RUN pip install -r requirements.txt

RUN mkdir locofy && cd locofy
COPY . /locofy/

WORKDIR /locofy

RUN aws configure set aws_access_key_id [PING_ME_FOR_ACCESS]
RUN aws configure set aws_secret_access_key [PING_ME_FOR_ACCESS]

ENV GIT_PYTHON_REFRESH=quiet
RUN dvc pull

EXPOSE 8003

CMD [ "python", "run.py", "--port", "8003" ]
