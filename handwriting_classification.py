import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
import time

"""
    This script contains basic implementations of several supervised 
    learning classification algorithms (random forests, naive Bayes, 
    K nearest neighbours (KNN), and a convolutional neural network). 
    
    The classifiers are used on a standard well-known dataset
    which contains simple representations of handwriting. 
    The goal of a classifier is to identify which digit in the range
    0-9 a datapoint represents. 
    
    The classifiers have not been optimised or fine-tuned (which is 
    bad since almost all default settings are terrible). Despite that,
    all of the approaches perform quite well in terms of precision, 
    recall and  accuracy. The best performing classifier (without tuning), 
    in terms of accuracy, turns out to be KNN. Given the simplicity of the
    data it is not unexpected that the basic KNN approach works well. 

    This is a simple analysis and does not include things like cross
    validation. Next steps: 1) try AutoML techniques such as in the tpot
    and auto-sklearn packages, 2) try XGBoost, and 3) manually tune 
    approaches (e.g., use a train, test, validate approach for all methods). 
    
"""



# First load the data:
dataset = datasets.load_digits()
labels = dataset.target
image_data = dataset.data

# Split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(image_data, 
                                                    labels, 
                                                    test_size=0.2, 
                                                    shuffle=True, 
                                                    stratify=labels,
                                                    random_state=2021)

##############################
### RANDOM FOREST APPROACH ###
##############################

tic = time.time()

# Build RF classifier:
forest = RandomForestClassifier(random_state=1)

# Fit RF classifier:
forest.fit(X_train, y_train)

# Obtain RF predictions:
y_predict = forest.predict(X_test)

rf_time = time.time() - tic

# Print out RF performance evaluation:
print("=====================================")
print("Random Forest Performance Evaluation")
print("=====================================")
print("Confusion matrix:\n")
print(confusion_matrix(y_test, y_predict))
print("------------------------------------")
print("Classification report:\n")
print(classification_report(y_test, y_predict))
print("\n")

############################
### NAIVE BAYES APPROACH ###
############################

tic = time.time()

# Build NB classifier:
nb = MultinomialNB()

# Fit RF classifier:
nb.fit(X_train, y_train)

# Obtain RF predictions:
y_predict = nb.predict(X_test)

nb_time = time.time() - tic


# Print out RF performance evaluation:
print("=====================================")
print("Naive Bayes Performance Evaluation")
print("=====================================")
print("Confusion matrix:\n")
print(confusion_matrix(y_test, y_predict))
print("------------------------------------")
print("Classification report:\n")
print(classification_report(y_test, y_predict))
print("\n")


#####################################
### K Nearest Neighbours APPROACH ###
#####################################

tic = time.time()

# Build KNN classifier:
neigh = KNeighborsClassifier(n_neighbors=5)

# Fit KNN classifier:
neigh.fit(X_train, y_train)

# Obtain KNN predictions:
y_predict = neigh.predict(X_test)

knn_time = time.time() - tic


# Print out RF performance evaluation:
print("===========================================")
print("K Nearest Neighbours Performance Evaluation")
print("===========================================")
print("Confusion matrix:\n")
print(confusion_matrix(y_test, y_predict))
print("------------------------------------")
print("Classification report:\n")
print(classification_report(y_test, y_predict))
print("\n")



####################
### CNN APPROACH ###
####################

# Hyperparameter that could be tuned:
BATCH_SIZE = 5

#Additional preprocessing is required for a CNN approach:
image_data_cnn = dataset.images

# Split the data into training and test sets:
X_train_cnn, X_test_cnn, y_train, y_test = train_test_split(image_data_cnn, 
                                                            labels, 
                                                            test_size=0.2, 
                                                            shuffle=True, 
                                                            stratify=labels,
                                                            random_state=2021)

tic = time.time()

v_pixels = X_train_cnn.shape[1]
h_pixels = X_train_cnn.shape[2]

X_train_cnn2 = X_train_cnn.reshape((X_train.shape[0], 
                                    v_pixels, 
                                    h_pixels, 1)).astype('float32')
X_test_cnn2 = X_test_cnn.reshape((X_test.shape[0], 
                                  v_pixels, 
                                  h_pixels, 1)).astype('float32')

y_train_cnn = np_utils.to_categorical(y_train)
y_test_cnn = np_utils.to_categorical(y_test)
num_classes = y_test_cnn.shape[1]

# Create model
cnn = Sequential()
# Define input layer
cnn.add(Conv2D(32, (3, 3), input_shape=(v_pixels, 
                                        h_pixels, 1), activation='relu'))
# Define hidden layers
cnn.add(MaxPooling2D())
cnn.add(Dropout(0.2))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
# Define output layer
cnn.add(Dense(num_classes, activation='softmax'))
# Compile model
cnn.compile(loss='categorical_crossentropy',
            optimizer='adam', metrics=['accuracy'])


# Fit the model
cnn.fit(X_train_cnn2, y_train_cnn, validation_data=(X_test_cnn2, y_test_cnn),
          epochs=10, batch_size=BATCH_SIZE, verbose=0)

# Obtain CNN predictions:
y_predict = np.argmax(cnn.predict(X_test_cnn2), axis=1)

cnn_time = time.time() - tic


# Print out CNN performance evaluation:
print("===================================================")
print("Convolutional Neural Network Performance Evaluation")
print("===================================================")
print("Confusion matrix:\n")
print(confusion_matrix(y_test, y_predict))
print("---------------------------------------------------")
print("Classification report:\n")
print(classification_report(y_test, y_predict))
print("\n")

# Compare accuracies of all approaches:
print("========================")
print("Comparison of accuracies")
print("========================")

print(tabulate([['Random forest', forest.score(X_test, y_test), rf_time], 
                ['Naive Bayes', nb.score(X_test, y_test), nb_time], 
               ['K Nearest neighbours', neigh.score(X_test, y_test), knn_time], 
                ['Convolution neural network', sum(y_predict==y_test)/len(y_test), cnn_time]], 
               headers=['Approach', 'Accuracy', 'CPU (secs)']))


# This is the output from running the code:

# =====================================
# Random Forest Performance Evaluation
# =====================================
# Confusion matrix:

# [[35  0  0  0  0  0  0  0  1  0]
#  [ 0 36  0  0  0  0  0  0  0  0]
#  [ 0  0 35  0  0  0  0  0  0  0]
#  [ 0  0  0 35  0  1  0  1  0  0]
#  [ 0  0  0  0 35  0  0  1  0  0]
#  [ 0  0  0  0  0 35  0  0  0  2]
#  [ 1  0  0  0  0  0 35  0  0  0]
#  [ 0  0  0  0  0  0  0 36  0  0]
#  [ 0  3  0  1  0  0  0  2 29  0]
#  [ 0  0  0  0  0  0  0  0  1 35]]
# ------------------------------------
# Classification report:

#               precision    recall  f1-score   support

#            0       0.97      0.97      0.97        36
#            1       0.92      1.00      0.96        36
#            2       1.00      1.00      1.00        35
#            3       0.97      0.95      0.96        37
#            4       1.00      0.97      0.99        36
#            5       0.97      0.95      0.96        37
#            6       1.00      0.97      0.99        36
#            7       0.90      1.00      0.95        36
#            8       0.94      0.83      0.88        35
#            9       0.95      0.97      0.96        36

#     accuracy                           0.96       360
#    macro avg       0.96      0.96      0.96       360
# weighted avg       0.96      0.96      0.96       360



# =====================================
# Naive Bayes Performance Evaluation
# =====================================
# Confusion matrix:

# [[35  0  0  0  1  0  0  0  0  0]
#  [ 0 29  2  0  0  0  2  0  1  2]
#  [ 0  2 30  0  0  0  0  0  3  0]
#  [ 0  0  1 31  0  0  0  2  0  3]
#  [ 0  0  0  0 35  0  0  0  1  0]
#  [ 0  0  0  0  0 32  0  0  0  5]
#  [ 0  0  0  0  1  0 35  0  0  0]
#  [ 0  0  0  0  0  0  0 36  0  0]
#  [ 0  5  0  0  0  0  0  1 27  2]
#  [ 0  1  0  0  0  0  0  2  2 31]]
# ------------------------------------
# Classification report:

#               precision    recall  f1-score   support

#            0       1.00      0.97      0.99        36
#            1       0.78      0.81      0.79        36
#            2       0.91      0.86      0.88        35
#            3       1.00      0.84      0.91        37
#            4       0.95      0.97      0.96        36
#            5       1.00      0.86      0.93        37
#            6       0.95      0.97      0.96        36
#            7       0.88      1.00      0.94        36
#            8       0.79      0.77      0.78        35
#            9       0.72      0.86      0.78        36

#     accuracy                           0.89       360
#    macro avg       0.90      0.89      0.89       360
# weighted avg       0.90      0.89      0.89       360



# ===========================================
# K Nearest Neighbours Performance Evaluation
# ===========================================
# Confusion matrix:

# [[36  0  0  0  0  0  0  0  0  0]
#  [ 0 36  0  0  0  0  0  0  0  0]
#  [ 0  0 35  0  0  0  0  0  0  0]
#  [ 0  0  0 36  0  0  0  1  0  0]
#  [ 0  0  0  0 36  0  0  0  0  0]
#  [ 0  0  0  0  0 36  0  0  0  1]
#  [ 0  0  0  0  0  0 36  0  0  0]
#  [ 0  0  0  0  0  0  0 36  0  0]
#  [ 0  1  0  1  0  0  0  0 33  0]
#  [ 0  0  0  1  0  0  0  0  1 34]]
# ------------------------------------
# Classification report:

#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        36
#            1       0.97      1.00      0.99        36
#            2       1.00      1.00      1.00        35
#            3       0.95      0.97      0.96        37
#            4       1.00      1.00      1.00        36
#            5       1.00      0.97      0.99        37
#            6       1.00      1.00      1.00        36
#            7       0.97      1.00      0.99        36
#            8       0.97      0.94      0.96        35
#            9       0.97      0.94      0.96        36

#     accuracy                           0.98       360
#    macro avg       0.98      0.98      0.98       360
# weighted avg       0.98      0.98      0.98       360



# ===================================================
# Convolutional Neural Network Performance Evaluation
# ===================================================
# Confusion matrix:

# [[36  0  0  0  0  0  0  0  0  0]
#  [ 0 36  0  0  0  0  0  0  0  0]
#  [ 0  0 35  0  0  0  0  0  0  0]
#  [ 0  0  0 36  0  0  0  1  0  0]
#  [ 0  0  0  0 35  0  0  0  0  1]
#  [ 0  0  0  0  0 35  0  0  0  2]
#  [ 1  0  0  0  0  0 35  0  0  0]
#  [ 0  0  0  0  0  0  0 36  0  0]
#  [ 0  1  0  0  0  0  0  1 30  3]
#  [ 0  0  0  0  0  0  0  0  0 36]]
# ---------------------------------------------------
# Classification report:

#               precision    recall  f1-score   support

#            0       0.97      1.00      0.99        36
#            1       0.97      1.00      0.99        36
#            2       1.00      1.00      1.00        35
#            3       1.00      0.97      0.99        37
#            4       1.00      0.97      0.99        36
#            5       1.00      0.95      0.97        37
#            6       1.00      0.97      0.99        36
#            7       0.95      1.00      0.97        36
#            8       1.00      0.86      0.92        35
#            9       0.86      1.00      0.92        36

#     accuracy                           0.97       360
#    macro avg       0.98      0.97      0.97       360
# weighted avg       0.98      0.97      0.97       360



# ========================
# Comparison of accuracies
# ========================
# Approach                      Accuracy    CPU (secs)
# --------------------------  ----------  ------------
# Random forest                 0.961111   0.266422
# Naive Bayes                   0.891667   0.000999212
# K Nearest neighbours          0.983333   0.018002
# Convolution neural network    0.972222   2.60167