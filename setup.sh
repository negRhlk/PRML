#!/bin/bash

# iris 
wget -P data/ https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data 

# part of mnist (only test data)
wget -P data/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -P data/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz