# PRML
Implementation for ML Algorithum in PRML 



## Setup 

Perhaps this will work for now though I plan to push the docker image. 

```

# clone repository and build docker image

git clone https://github.com/hedwig100/PRML && cd PRML
docker build -t  prml .
docker run -it --name prml -p 8888:8888 prml /bin/bash

# in docker container 
PRML/setup.sh # or cd PRML && ./setup.sh

```