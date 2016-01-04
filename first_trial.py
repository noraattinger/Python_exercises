
# coding: utf-8

# # TEST: Welcome to Azure ML Jupyter Notebooks!

# ![Jupyter-azureml](https://mysampledata.blob.core.windows.net/data/nb.studio.and.jupyter.png)

# 
# ### Jupyter notebooks is a Preview feature in ML Studio. With Jupyter notebooks you can:
# 
# * Edit code in the browser, with automatic syntax highlighting, indentation, and tab completion/introspection.
# * Run code from the browser, with the results of computations attached to the code which generated them.
# * See the results of computations with rich media representations, such as HTML, LaTeX, PNG, SVG, PDF, etc.
# 
# ### A hosted notebook service
# 
# * AzureML Studio and Jupyter notebooks can be run from any OS in any modern browser
# * Zero install! Full Python 2 and 3 support (more languages coming later)
# * Scalable service running on Azure (notebooks run Linux (Ubuntu) / Docker)
# 

# ##Jupyter Notebooks: a quick overview

# Notebooks are made up of a linear sequences of "Cells":
# 
# * Some cells are "code" cells and can be executed via selecting (double-clicking) the cell and pressing "Shift+Enter"
# * Some cells are "Markdown" cells and can be used for documentation
# 
# Edit vs command mode
# 
# * When you hit escape, the cell border turns grey - you are in "command" mode
# * When you double-click on a cell, the border turns green - you in "edit" mode
# 
# In the Edit mode, you can enter text or code and shift-enter to commit. In the command mode you can navigate and manipulate cells. You can also use the menus to add cells, delete cells, navigate up/down, run/stop, change cell types, etc. 
# 
# There are many shortcuts available. Some of these include:
# 
#  Basic navigation: up-arrow, down-arrow, enter, shift-enter, up/k, down/j
#  Saving the notebook: s
#  Cell types:trret y (code), m (markdown), 1-6 (heading level)
#  Add cells: a (cell above), b (cell below)
#  Cell editing: x (delete), c(copy)
# 
# For a full list, see Help/Keyboard shortcuts on the top Menu.
# 
# Now let's try some examples! **To run: double-click in the code cell below, then press Shift+Enter:**

# In[1]:

# This is a code cell. Click on it and press Shift+Enter. While it's executing, the prompt will turn into In [*]:

get_ipython().magic(u'pylab inline')
print "Python is easy to learn!"

plot(randn(100))


# ## Quick markdown overview: this is a Markdown cell. If you double-click it, you will see the raw markdown text
# 
# ### headings are denoted by "#". This line starts with "###"
# #### This one starts with "####"
# 
# * Bullets start with a "\*"
# * Another bullet line
# 
# > here's a block quote denoted by ">"
# 
# *Italic text text is surrounded by matching asterisks*
# 
# **Bold text is surrounded by double asterisks**
# 
# ```
# # you can use ``` to suspend formatting, for example for code:
# import azureml 
# print 42
# ```
# 
# http://www.example.com #urls are parsed automatically

# # Using Jupyter in AzureML
# 
# You can use Jupyter notebooks in various scenarios in AzureML:
# 
# * **A basic, blank notebook as a data playground**
# 
#  Simply do a +New and select a Blank Notebook. Use the full Anaconda distro to build up your notebook, import data from Azure, etc.
#  
# 
# * **For exploring Azure ML datasets and visualization**
# 
#  Select a Dataset and then click on "Explore in notebook" at the bottom of the page. 
#  
# 
# * **For exploring intermediate data in Experiments**
# 
#  In your Experiment, add a convert-to-csv node and then right click and select Open in notebook.
# 
# 
# * **Scratchpad for testing out python code for inclusion in Experiments**
# 
#  If you are writing a Python module, you can fire up a notebook, test your code snippet, then copy/paste it into the Python module in the Experiment
# 
# 
# ## Python Environment
# 
# Both the Jupyter notebooks and the AzureML execution environment are backed by the Anaconda Distro version 2.2 (64-bit). You can also install Anaconda on your local machine and use Notebooks or and IDE such as Python Tools for Visual Studio locally.  To see the full list of pkgs, see the Anaconda link at the bottom of this notebook.
# 
# In the notebooks you can use Python 2.x or Python 3.x.  *In the Studio you can only use Python 2.x currently*.
# 
# ## Saving and loading notebooks
# 
# Notebooks are automatically saved/checkpointed.  You can explicitly save your notebook by clicking the Save button on the Jupyter menu.  You can view the list of notebooks in the Studio, in the Notebooks tab.
# 
# To upload a notebook, you can either drag/drop it onto notebook surface or use the "Upload" button.
# 
# 
# ## Limitations
# 
# Currently network access is limited to Azure.  This means pip is not available (yet).
# 
# You can't upload text files, create folders or Terminals (yet).

# # Running code
# 
# 
# When you request a notebook, a *notebook server* is created for you and you are place in a fresh notebook.  You can access your notebook server by clicking on the larget "Jupyter" logo on the top left.  From the notebook server you can:
# 
# * Create more notebooks (they will show up in the Studio once saved)
# * Manipulate notebooks (rename, copy, delete, ...)
# * Upload local notebooks
# 
# When a notebook is created for you, your credentials are automatically routed into the newly created notebook.  This allows you for example to view, edit and save your datasets.  You can also explicitly copy/paste the access codes by selecting the menu item at the bottom of the page.  You may choose to do this when accessing datasets from a local IDE or notebook for example.
# 
# The primary packages that you generally need are:
# 
# * numpy # numerics
# * pandas # dataframes
# * matplotlib #visualization
# * scikit-learn # for ML (or statsmodel, pybrain, etc).
# * azureml # SDK for AzureML access
# 
# Here's a short example.  **To run: double-click in the code cell below, then press Shift+Enter:**

# In[ ]:

# to run this code, double-click in the cell, then Shift+Enter

import pandas as pd

#note: data has to reside on Azure 
input_data_url = 'https://mysampledata.blob.core.windows.net/data/text.csv'  # Iris dataset

text_frame = pd.read_csv(input_data_url,encoding='latin-1')
text_frame


# In[ ]:

# let's get some quick stats on the data

text_frame.describe()


# For further information on Pandas, see the links at the bottom of this notebook.

# ##Using the Azure ML Python Client Library (azureml)
# 
# The azureml library can be imported into your notebook or other Python IDE's and used to access your Experiments and datasets.
# 
# The azureml SDK is already pre-installed for you.
# 
# 

# In[ ]:

# First let's setup the connection and auth:

from azureml import Workspace

ws = Workspace(workspace_id='3064b77827bf495296f258c0aa5ff91f',
               authorization_token='df28e0ba5c7f4c699e30db772c042247',
               endpoint='https://studioapi.azureml.net' )


# To enumerate all the example Experiments:

# In[ ]:

for ex in ws.example_experiments:
    print (ex.description)


# To view all the datasets:

# In[ ]:

for ds in ws.datasets:
    print(ds.name)


# Or just the user-created datasets:
# 

# In[ ]:

for ds in ws.user_datasets:
    print(ds.name)


# Note: Jupyter has code completion! Try typing "ds." (without the quotes) while in Edit mode.  You'll get a pop up menu to select from:
# ![Jupyter-azureml](https://mysampledata.blob.core.windows.net/data/jupyter.isense.PNG)
# 
# Try it yourself by uncommenting the following line and after the ., hit Tab:

# In[ ]:

#ds.


# For full documentation and more exmaples, please see:
# 
# https://github.com/Azure/Azure-MachineLearning-ClientLibrary-Python

# 

# #Using scikit-learn and matplotlib
# 
# The Anaconda distro contains a rich set of pkgs for data analysis, machine learning and technical computing.  
# 
# scikit-learn is a popular pkg for data science.  Its website has a number of examples that you can try here.  Others include statsmodels and pybrain.
# 
# Below are a few matplotlib and scikit examples.
# 
# **For each of these self contained examples below, double-click in the cell and press Shift+Enter**
# 

# ##Matplotlib: histogram
# 
# Here's an example of building up a histogram graph

# In[ ]:

"""
Demo of the histogram (hist) function with a few features.

In addition to the basic histogram, this demo shows a few optional features:

    * Setting the number of data bins
    * The ``normed`` flag, which normalizes bin heights so that the integral of
      the histogram is 1. The resulting histogram is a probability density.
    * Setting the face color of the bars
    * Setting the opacity (alpha value).

"""
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


# example data
mu = 100 # mean of distribution
sigma = 15 # standard deviation of distribution
x = mu + sigma * np.random.randn(10000)

num_bins = 50
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()


# ### Scatter plot using Seaborn
# 
# Seaborn runs on top of matplotlib and provides a more modern look:

# In[ ]:

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")

# Generate a random dataset with strong simple effects and an interaction
n = 80
rs = np.random.RandomState(11)
x1 = rs.randn(n)
x2 = x1 / 5 + rs.randn(n)
b0, b1, b2, b3 = .5, .25, -1, 2
y = b0 + b1 * x1 + b2 * x2 + b3 * x1 * x2 + rs.randn(n)
df = pd.DataFrame(np.c_[x1, x2, y], columns=["x1", "x2", "y"])

# Show a scatterplot of the predictors with the estimated model surface
sns.interactplot("x1", "x2", "y", df)


# ### Fit and plot residuals

# In[ ]:

sns.set(style="whitegrid")

# Make an example dataset with y ~ x
rs = np.random.RandomState(7)
x = rs.normal(2, 1, 75)
y = 2 + 1.5 * x + rs.normal(0, 2, 75)

# Plot the residuals after fitting a linear model
sns.residplot(x, y, lowess=True, color="g")


# ### SVC / Digits
# 

# In[ ]:

"""
A recursive feature elimination example showing the relevance of pixels in
a digit classification task.

"""
print(__doc__)

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

# Plot pixel ranking
plt.matshow(ranking)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()


# In[ ]:

"""
=========================================================
Logistic Regression 3-class Classifier
=========================================================
Show below is a logistic-regression classifiers decision boundaries on the
`iris <http://en.wikipedia.org/wiki/Iris_flower_data_set>`_ dataset. The
datapoints are colored according to their labels.
"""
print(__doc__)


# Code source: Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()


# ### Confusion matrix

# In[ ]:

"""
================
Confusion matrix
================

Example of confusion matrix usage to evaluate the quality
of the output of a classifier on the iris data set. The
diagonal elements represent the number of points for which
the predicted label is equal to the true label, while
off-diagonal elements are those that are mislabeled by the
classifier. The higher the diagonal values of the confusion
matrix the better, indicating many correct predictions.

The figures show the confusion matrix with and without
normalization by class support size (number of elements
in each class). This kind of normalization can be
interesting in case of class imbalance to have a more
visual interpretation of which class is being misclassified.

Here the results are not as good as they could be as our
choice for the regularization parameter C was not the best.
In real life applications this parameter is usually chosen
using :ref:`grid_search`.

"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()


# # Resources
# 
# The following links provide further info about Jupyter and AzureML:
# 
# ### Jupyter
# 
# * Jupyter:  http://www.ipython.org
# * Notebook how to:  http://nbviewer.ipython.org/github/ipython/ipython/blob/3.x/examples/Notebook/Index.ipynb
# * Sample notebooks:  http://nbviewer.ipython.org 
# * Markdown:   https://help.github.com/articles/markdown-basics/ 
# 
# ### Azure ML
# 
# * AzureML Client SDK:   https://github.com/Azure/Azure-MachineLearning-ClientLibrary-Python (already installed)
# * Azure Python SDK:   http://azure-sdk-for-python.readthedocs.org/en/latest/ (already installed)
# 
# ### Python Tools for Visual Studio
# 
# * PTVS:   https://www.visualstudio.com/en-us/explore/python-vs 
# 
# ### Anaconda and key pkgs
# 
# 
# * Anaconda:   http://docs.continuum.io/anaconda/pkg-docs.html
# * Numpy:    http://docs.scipy.org/doc/numpy/reference/
# * Pandas:   http://pandas.pydata.org/pandas-docs/stable/
# * SciPy:   http://docs.scipy.org/doc/scipy/reference/
# * Matplotlib:   http://matplotlib.org/contents.html
# 
# ### Credits
# 
# * Jupyter:   Fernando, Brian, Min, Kyle, ... and the whole Jupyter team
# * Examples:   scikit-learn and matplotlib websites (Olivier, John, ...)
# * Anaconda:   Travis, Peter and the Continuum team
# 

# ## Questions?
# 
# * If you have any questions, issues, bugs (repro appreciated!), ... please send mail to **nbhelp@microsoft.com**
# * If you have examples, notebooks, etc. that you'd like to add to our gallery, please let us know!
# 
# 
# ## Help us improve!
# 
# * Please take this 2 minute survey and let us know what you'd like us to work on:
# 
#     http://surveymonkey.com/s/JupyterOnAzureml 
# 
# 
# Thanks & enjoy!

# In[ ]:



