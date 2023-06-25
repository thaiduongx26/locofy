FROM python:3.9-slim

COPY requirements.* ./
RUN pip install -r requirements.txt

RUN mkdir locofy && cd locofy
COPY . /locofy/

RUN dvc pull -f

WORKDIR /locofy

CMD [ "python", "run.py", "--port", "4000" ]
