FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

WORKDIR /home/azureuser

RUN git clone -b huinan/synthetic-AML-feedback https://github.com/microsoft/dp-transformers.git

RUN ls -lah /home/azureuser

RUN conda create --name myenv python=3.9.12

WORKDIR /home/azureuser/dp-transformers
RUN /opt/miniconda/envs/myenv/bin/pip install numpy
RUN /opt/miniconda/envs/myenv/bin/pip install -r requirements.txt
RUN /opt/miniconda/envs/myenv/bin/pip install .
