# PRML

In this repository, various machine learning algorithms in 「Pattern Recognition and Machine Learning」without using machine learning packages such as `scikit-learn`. 

All the algorithms are implemeted in the directory `prml`. In `short_notebook` directory, I use `prml` package (which I make myself). This notebooks are short-version, so 
this is useful when you want to see how these algorithms work. <br>

However, many people must want to know how these algorithms are implemented. These notebooks in `notebook` directory 
meet this need. In `notebook`, nothing of `prml` is used. All algorithms are implemented in almost the same way as `prml`. You can write and modify the code in jupyter notebook. 

**Notebooks**

- [chapter2 probability distributions](https://github.com/hedwig100/PRML/blob/main/notebook/chapter02_probability_distributions.ipynb)<br>
- [chapter3 linear regression](https://github.com/hedwig100/PRML/blob/main/notebook/chapter03_linear_regression.ipynb)<br>
- [chapter4 linear classifier](https://github.com/hedwig100/PRML/blob/main/notebook/chapter04_linear_classifier.ipynb)<br>
- [chapter5 neural network](https://github.com/hedwig100/PRML/blob/main/notebook/chapter05_neural_network.ipynb)<br>
- [chapter6 kernel methods](https://github.com/hedwig100/PRML/blob/main/notebook/chapter06_kernel_methods.ipynb)<br>
- [chapter7 sparse_kernel_machine](https://github.com/hedwig100/PRML/blob/main/notebook/chapter07_sparse_kernel_machines.ipynb)<br>
- [chapter8 graphical models](https://github.com/hedwig100/PRML/blob/main/notebook/chapter08_graphical_models.ipynb)<br>
- [chapter9 mixture models](https://github.com/hedwig100/PRML/blob/main/notebook/chapter09_mixture_models.ipynb)<br>
- [chapter10 approximate inference](https://github.com/hedwig100/PRML/blob/main/notebook/chapter10_approximate_inference.ipynb)<br>
- [chapter11 sampling methods](https://github.com/hedwig100/PRML/blob/main/notebook/chapter11_sampling_methods.ipynb)<br>
- [chapter12 continuous latent variables](https://github.com/hedwig100/PRML/blob/main/notebook/chapter12_continuous_latent_variables.ipynb)<br>
- [chapter13 sequential data](https://github.com/hedwig100/PRML/blob/main/notebook/chapter13_sequential_data.ipynb)<br>
- [chapter14 combining_models](https://github.com/hedwig100/PRML/blob/main/notebook/chapter14_combining_models.ipynb)<br>
<br>

**Short Notebooks**

- [chapter2 probability distributions(short ver)](https://github.com/hedwig100/PRML/blob/main/short_notebook/chapter02_short_ver.ipynb)<br>
- [chapter3 linear regression(short ver)](https://github.com/hedwig100/PRML/blob/main/short_notebook/chapter03_short_ver.ipynb)<br>
- [chapter4 linear classifier(short ver)](https://github.com/hedwig100/PRML/blob/main/short_notebook/chapter04_short_ver.ipynb)<br>
- [chapter5 neural network(short ver)](https://github.com/hedwig100/PRML/blob/main/short_notebook/chapter05_short_ver.ipynb)<br>
- [chapter6 kernel methods(short ver)](https://github.com/hedwig100/PRML/blob/main/short_notebook/chapter06_short_ver.ipynb)<br>
- [chapter7 sparse_kernel_machine(short ver)](https://github.com/hedwig100/PRML/blob/main/short_notebook/chapter07_short_ver.ipynb)<br>
- [chapter8 graphical models(short ver)](https://github.com/hedwig100/PRML/blob/main/short_notebook/chapter08_short_ver.ipynb)<br>
- [chapter9 mixture models(short ver)](https://github.com/hedwig100/PRML/blob/main/short_notebook/chapter09_short_ver.ipynb)<br>
- [chapter10 approximate inference(short ver)](https://github.com/hedwig100/PRML/blob/main/short_notebook/chapter10_short_ver.ipynb)<br>
- [chapter11 sampling methods(short ver)](https://github.com/hedwig100/PRML/blob/main/short_notebook/chapter11_short_ver.ipynb)<br>
- [chapter12 continuous latent variables(short ver)](https://github.com/hedwig100/PRML/blob/main/short_notebook/chapter12_short_ver.ipynb)<br>
- [chapter13 sequential data(short ver)](https://github.com/hedwig100/PRML/blob/main/short_notebook/chapter13_short_ver.ipynb)<br>
- [chapter14 combining_models(short ver)](https://github.com/hedwig100/PRML/blob/main/short_notebook/chapter14_short_ver.ipynb)<br>

## Setup 

There are three suggested ways to set up the environment to run this python repository. There are many other ways to set up, but Example2 may be 
the best way (because it doesn't take so much time to build docker image)<br>
  
**Example1** <br>
You can clone this repository and make virtual environment. <br>

```

# clone repository and make venv 

git clone https://github.com/hedwig100/PRML && cd PRML
chmod 755 setup.sh && ./setup.sh
python3 -m venv prml_venv 

source prml_venv/bin/activate
pip install -r requirements.txt # in virtual environment 
pip install notebook

```
<br>

**Example2** <br>
You can clone this repository and build docker image. 
You need to have docker installed. <br>

```

# clone repository and build docker image
git clone https://github.com/hedwig100/PRML && cd PRML
docker build -t prml .
docker run -it --name prml -p 8888:8888 prml /bin/bash

# in docker container 
cd PRML && ./setup.sh

```
in docker container, you can use `jupyter notebook` command. <br> 
<br>

**Example3** <br> 
You can pull docker image from docker hub. [This image](https://hub.docker.com/r/hedwig100/prml) is uploaded to docker hub. <br>
In this way, you cannot see document because no document is in the docker container. 

```

# pull docker image and run
docker pull hedwig100/prml:latest
docker run -it --name prml -p 8888:8888 hedwig100/prml:latest /bin/bash

# in docker container 
cd PRML && ./setup.sh

```
in docker container, you can use `jupyter notebook` command. <br> 
