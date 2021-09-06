FROM jupyter/base-notebook:hub-1.4.2

ARG USER="prmluser"
ARG UID="1001"
# same gid as jovyan
ARG GID="100" 

USER root

RUN useradd ${USER} -u ${UID} -g ${GID}
RUN mkdir -p /home/${USER}/PRML && \
    fix-permissions "/home/${USER}"

ENV HOME="/home/${USER}" \ 
    PYTHONPATH="$PYTHONPATH:/home/${USER}/PRML"

COPY . /home/${USER}/PRML/
RUN pip install -r /home/${USER}/PRML/requirements.txt

USER ${UID}
WORKDIR ${HOME}