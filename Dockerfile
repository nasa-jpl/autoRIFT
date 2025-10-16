FROM continuumio/miniconda3:main

WORKDIR /project

COPY . .

RUN conda env update -n base --file environment.yml --prune

RUN python -m pip install --use-pep517 --no-build-isolation -e .