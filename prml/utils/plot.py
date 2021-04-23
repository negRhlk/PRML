"""Plot
"""


import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from prml.utils.encoder import OnehotToLabel 

color = ["red","blue","lightgreen","yellow","orange","purple","pink"] 

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