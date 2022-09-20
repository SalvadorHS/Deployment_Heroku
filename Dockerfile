FROM continuumio/anaconda3
COPY . /home/Streamlit_app
WORKDIR /home/Streamlit_app

RUN pip install flasgger
RUN pip install flask
RUN pip install numpy
RUN pip install pandas
RUN pip install scikit_learn
RUN pip install --ignore-installed streamlit

EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
