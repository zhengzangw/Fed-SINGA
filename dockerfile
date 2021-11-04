FROM zhengzangw/fedsinga:latest

# The above docker-image is composed of following layers
# FROM continuumio/miniconda3:4.10.3

# RUN conda config --set always_yes yes --set changeps1 no
# RUN conda install python=3.6
# RUN conda install -c nusdbsystem -c conda-forge singa=3.1.0=cpu_py36

# COPY requirements.txt .
# RUN pip install -r requirements.txt
# WORKDIR /app
