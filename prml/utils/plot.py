"""Plot

    plot_regressionr1D 
    plot_classifier
"""

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from prml.utils.encoder import OnehotToLabel 

color = ["red","blue","lightgreen","yellow","orange","purple","pink"] 
cmaps = [[0.122, 0.467, 0.706],"orange","green"]

def plot_regression1D(X_tr,y_tr,regressor,title,f,lower = 0,upper = 2*np.pi):
    """plot regressor 

    Args:
        X_tr (1-D array) : training data,explanatory variable 
        y_tr (1-D array) : training data,target variable 
        regressor (object) : trained regressor model which must have "predict" method 
        title (string) : title of the plot 
        f (function) : regression function 
        lower,upper (float) : lower <= x <= upper
    """
    X = np.linspace(lower,upper,100).reshape(-1,1)
    y_pred = regressor.predict(X) 
    y_true = f(X)
    
    rmse = np.mean((y_true - y_pred)**2)**0.5
    print(f"RMSE : {rmse}")
    
    fig,ax = plt.subplots(1,1,figsize = (10,7))
    ax.plot(X,y_pred,label="Predict",color=cmaps[0])
    ax.plot(X,y_true,label="Ground Truth",color=cmaps[1])
    ax.scatter(X_tr,y_tr,label="Training Data",color=cmaps[2])
    ax.set_title(title)
    
    plt.legend()
    plt.show()

def plot_regression1D_with_std(X_tr,y_tr,regressor,title,f,lower = 0,upper = 2*np.pi):
    """plot regressor 

    Args:
        X_tr (1-D array) : training data,explanatory variable 
        y_tr (1-D array) : training data,target variable 
        regressor (object) : trained regressor model which must have "predict" method which should have 'return_std = True' as parameter 
        title (string) : title of the plot 
        f (function) : regression function 
        lower,upper (float) : lower <= x <= upper
    """
    X = np.linspace(lower,upper,100).reshape(-1,1)
    y_pred,y_std = regressor.predict(X,return_std=True)
    y_true = f(X)
    
    rmse = np.mean((y_pred - y_true)**2)**0.5
    print(f"RMSE : {rmse}")
    
    fig,ax = plt.subplots(1,1,figsize = (10,7))
    ax.plot(X,y_pred,label="Predict",color=cmaps[0])
    
    y_pred_upper = y_pred + y_std
    y_pred_lower = y_pred - y_std 
    ax.fill_between(X.ravel(),y_pred_lower.ravel(),y_pred_upper.ravel(),alpha=0.3,color=cmaps[0])
    
    ax.plot(X,y_true,label="Ground Truth",color=cmaps[1])
    ax.scatter(X_tr,y_tr,label="Training Data",color=cmaps[2])
    ax.set_title(title)
    
    plt.legend()
    plt.show()


def plot_classifier(X_tr,y_tr,classifier,title=""):
    """plot classifier 

    Args:
        X_tr (2-D array) : training data 
        y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded 
        classifier (object) : trained classifier 
        title (str) : title of the plot
    """
    if y_tr.ndim == 2:
        transform = OnehotToLabel()
        y_tr = transform.fit_transform(y_tr) 
    cmap = ListedColormap(color[:len(np.unique(y_tr))])

    # prepare data 
    x_min,y_min = X_tr.min(axis = 0)
    x_max,y_max = X_tr.max(axis = 0) 
    x_min,y_min = x_min-0.1,y_min-0.1
    x_max,y_max = x_max+0.1,y_max+0.1
    x = np.linspace(x_min,x_max,100)
    y = np.linspace(y_min,y_max,100) 
    xs,ys = np.meshgrid(x,y)

    # predict 
    labels = classifier.predict(np.array([xs.ravel(),ys.ravel()]).T)
    if labels.ndim == 2:
        labels = transform.transform(labels) 
    labels = labels.reshape(xs.shape)

    # plot 
    figure,axes = plt.subplots(1,1,figsize=(10,7))
    axes.contourf(xs,ys,labels,alpha=0.3,cmap=cmap)
    axes.set_xlim(x_min,x_max)
    axes.set_ylim(y_min,y_max)
    for idx,label in enumerate(np.unique(y_tr)):
        axes.scatter(x=X_tr[y_tr == label,0],
                    y=X_tr[y_tr == label,1],
                    alpha=0.8,
                    c=color[idx],
                    label=label)
    axes.set_title(title)
    plt.legend()
    plt.show()