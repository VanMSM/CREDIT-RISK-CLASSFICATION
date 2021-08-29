# CREDIT RISK CLASSFICATION ANALYSIS

A SUPERVISED LEARNING PROJECT
___

## OVERVIEW OF THE ANALYSIS

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this project, I used  various techniques to train and evaluate models with imbalanced classes. I used a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.
___

### FINANCIAL INFORMATION

I used a historical dataset that includes the following:
* Loan Size
* Interest Rate
* Borrower Income
* Debt to Income
* Number of Accounts 
* Derogatory Marks
* Total Debt
* Loan Status

___

### WHAT THE MODEL IS PREDICTING

The model is created to predict the healthy loans versus the high risk loans to better identify the credit worthiness of customers.

___

### STAGES OF MACHINE LEARNING:

* Split the Data into Training and Testing Sets

* Created a Logistic Regression Model with the Original Data

* Predict a Logistic Regression Model with Resampled Training Data 


I used a logistic regression model to compare two versions of the dataset. First, I used the original dataset. Second, I resampled the data by using the RandomOverSampler module from the imbalanced-learn library.

For both cases, I got the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and lastly, generated a classification report.

---

### RESULTS

<img width="782" alt="Screen Shot 2021-08-28 at 8 27 56 PM" src="https://user-images.githubusercontent.com/80144026/131237298-1ccfc1b8-d4a4-4b8e-a49c-6d7a359b9eaa.png">



---

## Technologies

This project is written in Python with the following packages:

* imbalanced-learn
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

*  Imbalanced Learn using Conda. Best practice, use an environment rather than install in the base env

```python
conda activate dev
conda install -c conda-forge imbalanced-learn
conda list imbalanced-learn
```

*  PyDotPlus using Conda. Best practice, use an environment rather than install in the base env

```python
conda activate dev
conda install -c conda-forge pydotplus
conda list pydotplus
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
