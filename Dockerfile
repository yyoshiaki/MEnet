FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

COPY . /menet

WORKDIR /menet

RUN pip install pandas matplotlib seaborn scikit-learn optuna pyparsing==2.4.7\
    && python setup.py install

RUN apt-get update && apt-get install bedtools

CMD MEnet