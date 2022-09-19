FROM continuumio/anaconda3
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/

RUN pip install --upgrade pip
RUN pip install flasgger
RUN pip install flask
RUN pip install numpy
RUN pip install pandas
RUN pip install scikit_learn
RUN pip install --ignore-installed streamlit

CMD python app.py
