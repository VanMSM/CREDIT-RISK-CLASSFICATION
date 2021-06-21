# CREDIT RISK CLASSFICATION

A SUPERVISED LEARNING PROJECT

## Background

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this project, I used  various techniques to train and evaluate models with imbalanced classes. I used a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

Using imbalanced-learn library, I used a logistic regression model to compare two versions of the dataset. First, I used the original dataset. Second, I resampled the data by using the RandomOverSampler module from the imbalanced-learn library.

For both cases, I got the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and lastly, generated a classification report.

---

## Technologies

This project is written in Python with the following packages:

* imbalance-learn
* PyDotPlus
* Numpy
* Pandas

---

## Installation Guide

Before running the application first install and verify the following dependencies:

* Pandas from PyPI
```python
pip install pandas
```
* Numpy from PyPI
```python
pip install numpy
```

*  Imbalance Learn using Conda. Best practice, use an environment rather than install in the base env

```python
conda activate dev
conda install -c conda-forge imbalanced-learn
conda list imbalanced-learn
```

*  PyDotPlus using Conda. Best practice, use an environment rather than install in the base env

```python
conda activate dev
conda install -c conda-forge pydotplus
conda list imbalanced-learn
``` 

Next, Import the Modules

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

import warnings
warnings.filterwarnings('ignore')
```

Clone to your repo and run

---



## Contributors

[Van Miller Sarcov Maquilan](https://www.linkedin.com/in/van-miller-sarcov-maquilan-20b472202/) 


---

## License

Feel free to add to this code, and use for your own project. :)
