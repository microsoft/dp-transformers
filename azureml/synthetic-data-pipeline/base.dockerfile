FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

WORKDIR /home/azureuser

RUN git clone -b amanoel/organize-components https://github.com/microsoft/dp-transformers.git
RUN ls -lah /home/azureuser

RUN conda create --name myenv python=3.9.12
RUN /opt/miniconda/envs/myenv/bin/pip install --progress-bar off \
    numpy \
    pandas==1.5.3 \
    scikit-learn \
    shrike \
    torch==1.12.1 \
    transformers==4.20.1 \
    datasets==2.0.0 \
    opacus==1.1.3 \
    prv-accountant==0.1.0

WORKDIR /home/azureuser/dp-transformers
RUN /opt/miniconda/envs/myenv/bin/pip install --progress-bar off .