import matplotlib.pyplot as plt
from matplotlib import image
import pandas as pd
from matplotlib import colors
from IPython.display import display
import numpy as np
import textwrap

# Pandas used for plotting transferability matrix. 
# Pandas: © Copyright 2008-2022, the pandas development team.
# For more information pls visit:https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
# MathPlot lib used for plotting images.
# MathPlot: © Copyright 2002 - 2012 John Hunter, Darren Dale, Eric Firing, Michael Droettboom and the Matplotlib development team; 2012 - 2021 The Matplotlib development team.
# For more insformation pls visit https://matplotlib.org/stable/api/index

'''
Use matplotlib imshow to plot a single image.

args:
data: a single image numpy array (for instance, [28,28] for a MNIST image)
for more details on image plots please follow the matplot lib guide: https://matplotlib.org/3.5.1/tutorials/introductory/images.html
'''
def plot_image(data):
    #fig = plt.figure(figsize=(15, 7))
    #plt.title("Adversial examples")
    plt.imshow(data,cmap=plt.cm.gray, interpolation='nearest')
    plt.show() 

    
'''
Use matplotlib imshow to plot multiple images.

args:
data: image numpy array of multiple image (for instance,[100,28,28] for an MNIST image)
'''    
def plot_images(data):
    # plot the images: each image is 28x28 pixels
    # set up the figure
    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    #for now, only show top 50
    if(data.shape[0]<=50):
        total_number = data.shape[0]
    else:
        total_number = 50
    for i in range(total_number):
        #ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
       
        #here the interpolation controls "anti-aliasing". For our case this should be strictly "none" or "nearest" because
        #we dont need antiliasing since most of our decisions on selecting hyperparameters are based on visual clues
        #ax.imshow(data[i].reshape((32,32,3)), interpolation='nearest')
        ax.imshow(data[i].reshape((28,28)),cmap=plt.cm.gray, interpolation='nearest')
        #ax.text(0, 7, i, color='red', weight="bold")
        #ax.imshow(data[i].reshape((28,28)),cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


def plot_cifar_images(data):
    # plot the images: each image is 28x28 pixels
    # set up the figure
    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    #for now, only show top 50
    if(data.shape[0]<=50):
        total_number = data.shape[0]
    else:
        total_number = 50
    for i in range(total_number):
        #ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
       
        #here the interpolation controls "anti-aliasing". For our case this should be strictly "none" or "nearest" because
        #we dont need antiliasing since most of our decisions on selecting hyperparameters are based on visual clues
        ax.imshow(data[i].reshape((32,32,3)), interpolation='nearest')
        #ax.imshow(data[i].reshape((28,28)),cmap=plt.cm.gray, interpolation='nearest')
        #ax.text(0, 7, i, color='red', weight="bold")
        #ax.imshow(data[i].reshape((28,28)),cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    
'''
Use matplotlib imshow to plot multiple images.

args:
data: image numpy array of multiple image (for instance,[100,28,28] for an MNIST image)
'''    
def plot_images_labels(images, labels, label_index, num=10, title="", show_title=False):
    
    if len(images) >= 50:
        num_images = 20
    else:
        num_images = len(images)
        
    num = min(num, num_images)
            
    num_cols = 8
    num_rows = (num_images + num_cols - 1) // num_cols
    
    plt.figure(figsize=(15, 7))
    
    for i in range(num):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        if show_title:
            #title_text= title+str(label_index[labels[i]])
            #wrapped_tite = textwrap.fill(title_text, width=20)
            plt.title(title+str(label_index[labels[i]]))
        #else:
        #    plt.title(title)
        plt.axis('off')
    
    #plt.tight_layout()
    plt.show()
 
    
'''
Use matplotlib imshow to plot multiple images.

args:
data: image numpy array of multiple image (for instance,[100,28,28] for an MNIST image)
'''    
def plot_images_labels_by_index(images, labels, indices, label_index, num_images_to_display=10, title=""):    
    
    if len(indices) >= 50:
        num_images = 20
    else:
        num_images = len(indices)
    
    
    #num_images = len(indices)
    num_images_to_display = min(num_images_to_display, num_images)
    num_cols = 8
    num_rows = (num_images + num_cols - 1) // num_cols
    
    plt.figure(figsize=(15, 7))
    
    for i in range(num_images_to_display):
        idx = indices[i]
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[idx], cmap='gray')
        #plt.title(title+str(label_index[labels[i]]))
        plt.title(title+str(label_index[labels[idx]]))
        plt.axis('off')
    
    #plt.tight_layout()
    plt.show()

'''
Plot the pandas dataframe with gradients. The dataframe here is the transferability data obtained from the experiments.
For more details on background gradients please refer to pandas manual at: https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.background_gradient.html

This example is referred from stackoverflow: https://stackoverflow.com/questions/38931566/pandas-style-background-gradient-both-rows-and-columns
The function overrides pandas "background" function: https://github.com/pandas-dev/pandas/blob/29d81f3df81eb0a4d077ae1317df74d509cdc446/pandas/formats/style.py#L817
'''
def plot_data_points(data_frame):
    df = pd.DataFrame(data_frame)
    pd.set_option("display.precision", 3)
    df.columns= ["Float", "1" ,"2", "4", "8", "12", "16"]
    df.index = ["Float", "1", "2", "4", "8", "12", "16"]
    #visualize_data.plotTable(df)
    # generate some example data
    def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
        rng = M - m
        norm = colors.Normalize(m - (rng * low),
                                M + (rng * high))
        normed = norm(s.values)
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        return ['background-color: %s' % color for color in c]

    display(df.style.apply(background_gradient,
                cmap='PuBu',
                m=df.min().min(),
                M=df.max().max(),
                low=0,
                high=0.2))

'''    
Display transferability matrix with average of the entire row at the end.
This is a slight modification of the function "plot_data_points(dataframe)" of this class.
'''   
def plot_data_points_with_average(data_frame):
    df = pd.DataFrame(data_frame)
    pd.set_option("display.precision", 3)
    df.columns= ["Float", "1" ,"2", "4", "8", "12", "16", "Average"]
    df.index = ["Float", "1", "2", "4", "8", "12", "16"]
    #visualize_data.plotTable(df)

    def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
        rng = M - m
        norm = colors.Normalize(m - (rng * low),
                                M + (rng * high))
        normed = norm(s.values)
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        return ['background-color: %s' % color for color in c]

    display(df.style.apply(background_gradient,
                cmap='PuBu',
                m=df.min().min(),
                M=df.max().max(),
                low=0,
                high=0.2))

'''
Same as plot_data_points_with_average but to make the labels look good, rows and column label backgrounds are made a bit consistent in color.
For more details and examples on this, please follow: https://pandas.pydata.org/docs/user_guide/style.html
'''
def plot_data_points_with_average_stylized(data_frame):
    df = pd.DataFrame(data_frame)
    pd.set_option("display.precision", 3)
    df.columns= ["FP", "1" ,"2", "4", "8", "12", "16", "Average"]
    df.index = ["FP", "1", "2", "4", "8", "12", "16"]
    
    def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
        rng = M - m
        norm = colors.Normalize(m - (rng * low),
                                M + (rng * high))
        normed = norm(s.values)
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        return ['background-color: %s' % color for color in c]

    th_props = [
                ('font-weight', 'bold'),
                ('background-color', '#f7f7f9')
                ]

    # Set table styles
    styles = [
        dict(selector="th", props=th_props)
        ]

    display(df.style.apply(background_gradient,
                cmap='PuBu',
                m=df.min().min(),
                M=df.max().max(),
                low=0,
                high=0.2).set_table_styles(styles))
