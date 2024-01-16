FROM rayproject/ray:latest-gpu
#continuumio/miniconda3:23.10.0-1
#23.10.0-1 
#23.5.2-0-alpine  # bug: no apt package installation mechanism here ... 

WORKDIR /home/ray

COPY scripts/* .

# # Create the environment:
# COPY environment.yml .
# RUN conda env create -f environment.yml

# RUN apt update && apt upgrade -y && apt install -y rsync

# RUN conda create -y -n "ray" python=3.8.5

# # Make RUN commands use the new environment: equivalent to "conda activate ray"
# SHELL ["conda", "run", "-n", "ray", "/bin/bash", "-c"] 

RUN sudo apt update
RUN sudo apt install --yes ffmpeg      

# RUN python -m pip install --upgrade pip 
# RUN pip install -U ray[default]==2.9.0
RUN pip install -U azure-cli-core==2.29.1 azure-identity==1.7.0 azure-mgmt-compute==23.1.0 azure-mgmt-network==19.0.0 azure-mgmt-resource==20.0.0 msrestazure==0.6.4 
RUN pip install -U torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 'pydantic<2' 
RUN pip install -U transformers==4.32.1 opencv-python==4.7.0.72 ffmpeg-python==0.2.0 moviepy==1.0.3 pymilvus==2.2.11 sentence-transformers==2.2.2
RUN pip install -U 'pydantic<2'
