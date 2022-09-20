FROM continuumio/anaconda3
COPY . /home/Flask_app
WORKDIR /home/Flask_app

RUN pip install --upgrade pip
RUN pip install flasgger
RUN pip install matplotlib
RUN pip install flask
RUN pip install numpy
RUN pip install pandas
RUN pip install scikit_learn
RUN pip install gunicorn
RUN pip install Jinja2

EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app


