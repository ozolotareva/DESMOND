FROM ubuntu:16.04
RUN apt update && apt install -y python2.7 python-pip python-tk
RUN pip install networkx==1.10 numpy==1.15.4 scipy==1.1.0 scikit-learn==0.19.1 pandas==0.23.0 
RUN pip install fisher==0.1.5 matplotlib==2.2.3 gseapy==0.9.9 seaborn==0.9.0

COPY . /code
CMD bash /data/DESMOND/run_DESMOND.sh
