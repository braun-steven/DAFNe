FROM nvcr.io/nvidia/pytorch:21.02-py3

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build && \
  rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# Install dependencies
RUN pip install tensorboard

RUN pip install 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo

# Set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Pascal;Kepler+Tesla;Volta;Turing;Ampere"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Locally install detectron2
RUN pip install -e detectron2_repo

# Install DOTA api
RUN DEBIAN_FRONTEND="noninteractive" sudo apt -y update
RUN DEBIAN_FRONTEND="noninteractive" sudo apt -y install swig
RUN git clone https://github.com/CAPTAIN-WHU/DOTA_devkit && \
    cd DOTA_devkit && \
    swig -c++ -python polyiou.i && \
    python setup.py install && \
    cd poly_nms_gpu && \
    python setup.py install

# Install additional requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# For debugging purposes
RUN pip install pdbpp ipython
COPY ./.pdbrc.py /.pdbrc.py

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/app/.torch/"
WORKDIR /app/dafne
