FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

WORKDIR /home/azureuser

RUN git clone -b huinan/synthetic-AML-feedback https://github.com/microsoft/dp-transformers.git

RUN ls -lah /home/azureuser
WORKDIR /home/azureuser/dp-transformers
RUN pip install -r requirements.txt
RUN pip install .
