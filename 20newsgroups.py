from google.colab import drive
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.notebook import tqdm
import pickle
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import re
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing, metrics, model_selection
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy

"""### Mount drive"""

drive.mount('/content/drive')

"""## CODE

### loading the data from the csv file from the drive
"""

df_train = pd.read_csv('/content/drive/MyDrive/NLP Project/data/Data_CSV/traindata.csv')
df_test = pd.read_csv('/content/drive/MyDrive/NLP Project/data/Data_CSV/testdata.csv')

"""### preprocessing

"""

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Preprocess Train data
df_train['text'] = df_train['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df_train['text'] = df_train['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df_train['text'] = df_train['text'].apply(lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split()))

# Preprocess Test data
df_test['text'] = df_test['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df_test['text'] = df_test['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df_test['text'] = df_test['text'].apply(lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split()))

"""### feature extraction

"""

# Vectorize text data with only unigrams
tfidf_vect = TfidfVectorizer(ngram_range=(1, 1))  # Using only unigrams
tfidf_vect.fit(df_train['text'])

# Encode labels
encoder = preprocessing.LabelEncoder()
encoder.fit(df_train['label'])  # Fit the encoder on training data labels

# Prepare data for training
X_train = tfidf_vect.transform(df_train['text'])
y_train = encoder.transform(df_train['label'])  # Transform training labels

# Apply tfidf to the test data with the fit on the train
X_test = tfidf_vect.transform(df_test['text'])
y_test = encoder.transform(df_test['label'])  # Transform test labels

"""## Training Section

### Naive_Bayes
"""

# Define Model
Naive_Bayes_model = MultinomialNB(alpha=0.01)
# Start time to calcualte the train time
Naive_Bayes_start_time = time.time()
# train model
Naive_Bayes_model.fit(X_train, y_train)
# Get the end time
Naive_Bayes_end_time = time.time()
Naive_Bayes_elapsed_time = Naive_Bayes_end_time - Naive_Bayes_start_time
# calc train accuracy
Naive_Bayes_train_predictions = Naive_Bayes_model.predict(X_train)
Naive_Bayes_train_accuracy = metrics.accuracy_score(Naive_Bayes_train_predictions, y_train)
# calc test accuracy
Naive_Bayes_test_predictions = Naive_Bayes_model.predict(X_test)
Naive_Bayes_test_accuracy = metrics.accuracy_score(Naive_Bayes_test_predictions, y_test)


print("Train Accuracy:", Naive_Bayes_train_accuracy)
print("Test Accuracy:", Naive_Bayes_test_accuracy)
print(f"Train time: {Naive_Bayes_elapsed_time} seconds")

conf_mat = confusion_matrix(y_test, Naive_Bayes_test_predictions)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

"""### SVM"""

# Define Model
SVM_model = SVC(kernel='linear', C=1.0)
# Start time to calcualte the train time
SVM_start_time = time.time()
# train model
SVM_model.fit(X_train, y_train)
# Get the end time
SVM_end_time = time.time()
SVM_elapsed_time = SVM_end_time - SVM_start_time
# calc train accuracy
SVM_train_predictions = SVM_model.predict(X_train)
SVM_train_accuracy = metrics.accuracy_score(SVM_train_predictions, y_train)
# calc test accuracy
SVM_test_predictions = SVM_model.predict(X_test)
SVM_test_accuracy = metrics.accuracy_score(SVM_test_predictions, y_test)


print("Train Accuracy:", SVM_train_accuracy)
print("Test Accuracy:", SVM_test_accuracy)
print(f"Train time: {SVM_elapsed_time} seconds")

conf_mat = confusion_matrix(y_test, SVMModel.test_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()