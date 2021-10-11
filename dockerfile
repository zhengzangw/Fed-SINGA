FROM continuumio/miniconda3:4.10.3
RUN conda config --set always_yes yes --set changeps1 no
RUN conda install python=3.6
RUN conda install -c nusdbsystem -c conda-forge singa=3.1.0=cpu_py36
WORKDIR /app
COPY . .
