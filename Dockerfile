FROM python:3.12

WORKDIR /opt/app

COPY . /opt/app/

RUN pip install -r /opt/app/requirements.txt

EXPOSE 5000
CMD [ "python", "/opt/app/api.py" ]