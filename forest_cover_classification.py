
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
import time

"""
    This script contains basic implementations of several supervised 
    learning classification algorithms (random forests, naive Bayes, 
    K nearest neighbours (KNN), and a convolutional neural network). 
    
    The classifiers have not been optimised or fine-tuned (which is 
    bad since almost all default settings are terrible). Despite that,
    all of the approaches except nairve Bayes perform reasonably well in 
    terms of precision, recall and  accuracy. The best performing classifier 
    (without tuning), in terms of accuracy, turns out to be random forests. 

    This is a simple analysis and does not include things like cross
    validation. Next steps: 1) try AutoML techniques such as in the tpot
    and auto-sklearn packages, 2) try XGBoost, and 3) manually tune 
    approaches (e.g., use a train, test, validate approach for all methods). 
    
"""



# First load the data:
raw_df = pd.read_csv('cover_data.csv')
df = raw_df.to_numpy()
y = df[:, -1]
X = df[:, 0:-1]

# Split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    shuffle=True, 
                                                    stratify=y,
                                                    random_state=2021)
# Scale the data
scaler1 = StandardScaler()
X_train_scale1 = scaler1.fit_transform(X_train)
X_test_scale1 = scaler1.transform(X_test)

# ##############################
# ### RANDOM FOREST APPROACH ###
# ##############################

tic = time.time()

# Build RF classifier:
forest = RandomForestClassifier(random_state=1)

# Fit RF classifier:
forest.fit(X_train_scale1, y_train)

# Obtain RF predictions:
y_predict = forest.predict(X_test_scale1)

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

# NB requires non-negative data so use a different scaler
scaler2 = MinMaxScaler()
X_train_scale2 = scaler2.fit_transform(X_train)
X_test_scale2 = scaler2.transform(X_test)

tic = time.time()

# Build NB classifier:
nb = MultinomialNB()

# Fit NB classifier:
nb.fit(X_train_scale2, y_train)

# Obtain NB predictions:
y_predict = nb.predict(X_test_scale2)

nb_time = time.time() - tic

# Print out NB performance evaluation:
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
neigh.fit(X_train_scale1, y_train)

# Obtain KNN predictions:
y_predict = neigh.predict(X_test_scale1)

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
### FFN APPROACH ###
####################

y_train_cnn = np_utils.to_categorical(y_train)
y_test_cnn = np_utils.to_categorical(y_test)
num_classes = y_test_cnn.shape[1]

# Hyperparameter that could be tuned:
BATCH_SIZE = 200

# y_train_ffn = np_utils.to_categorical(y_train)
# y_test_ffn = np_utils.to_categorical(y_test)
# num_classes = y_test_cnn.shape[1]

tic = time.time()

# Create model
ffn = Sequential()
# Define input layer
ffn.add(Dense(64,  input_dim=X.shape[1], activation='relu'))
# Define hidden layers
ffn.add(Dense(32, activation='relu'))
# ffn.add(Dropout(0.2))
# Define output layer
ffn.add(Dense(num_classes, activation='softmax'))
# Compile model
ffn.compile(loss='sparse_categorical_crossentropy',
            optimizer='adam', metrics=['accuracy'])


# Fit the model
ffn.fit(X_train_scale1, y_train, validation_data=(X_test_scale1, y_test),
        epochs=10, batch_size=BATCH_SIZE, verbose=1)

# Obtain FFN predictions:
y_predict = np.argmax(ffn.predict(X_test_scale1), axis=1)

ffn_time = time.time() - tic


# Print out CNN performance evaluation:
print("===================================================")
print("Feed-forward Neural Network Performance Evaluation")
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


print(tabulate([['Random forest', forest.score(X_test_scale1, y_test), rf_time], 
                ['Naive Bayes', nb.score(X_test_scale1, y_test), nb_time], 
                ['K Nearest neighbours', neigh.score(X_test_scale1, y_test), knn_time], 
                ['Feed-forward neural neural network', sum(y_predict==y_test)/len(y_test), ffn_time]], 
                headers=['Approach', 'Accuracy', 'CPU (secs)']))


# This is the output from running the code:

    
# =====================================
# Random Forest Performance Evaluation
# =====================================
# Confusion matrix:

# [[39975  2299     1     0    12     2    79]
#  [ 1232 55163   114     0    77    56    19]
#  [    1   101  6882    23     8   136     0]
#  [    0     0    55   475     0    19     0]
#  [   29   369    14     0  1478     9     0]
#  [    5   105   232    20     5  3106     0]
#  [  185    29     0     0     0     0  3888]]
# ------------------------------------
# Classification report:

#               precision    recall  f1-score   support

#            1       0.96      0.94      0.95     42368
#            2       0.95      0.97      0.96     56661
#            3       0.94      0.96      0.95      7151
#            4       0.92      0.87      0.89       549
#            5       0.94      0.78      0.85      1899
#            6       0.93      0.89      0.91      3473
#            7       0.98      0.95      0.96      4102

#     accuracy                           0.95    116203
#    macro avg       0.95      0.91      0.93    116203
# weighted avg       0.96      0.95      0.95    116203



# =====================================
# Naive Bayes Performance Evaluation
# =====================================
# Confusion matrix:

# [[19798 21579    50     0     0     0   941]
#  [ 8641 45740  2179     1     0    10    90]
#  [    0   394  6680    35     0    42     0]
#  [    0     0   431    78     0    40     0]
#  [  173  1536   190     0     0     0     0]
#  [   30   814  2512     5     0   112     0]
#  [ 1260   551    20     0     0     0  2271]]
# ------------------------------------
# Classification report:

#               precision    recall  f1-score   support

#            1       0.66      0.47      0.55     42368
#            2       0.65      0.81      0.72     56661
#            3       0.55      0.93      0.70      7151
#            4       0.66      0.14      0.23       549
#            5       0.00      0.00      0.00      1899
#            6       0.55      0.03      0.06      3473
#            7       0.69      0.55      0.61      4102

#     accuracy                           0.64    116203
#    macro avg       0.54      0.42      0.41    116203
# weighted avg       0.64      0.64      0.62    116203



# C:\Users\brend\anaconda3\envs\test7\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, msg_start, len(result))
# C:\Users\brend\anaconda3\envs\test7\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, msg_start, len(result))
# C:\Users\brend\anaconda3\envs\test7\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, msg_start, len(result))
# ===========================================
# K Nearest Neighbours Performance Evaluation
# ===========================================
# Confusion matrix:

# [[39216  2923     1     0    30     5   193]
#  [ 2349 53736   167     0   213   169    27]
#  [    4   188  6539    47    17   356     0]
#  [    0     0   109   403     0    37     0]
#  [   50   347    21     0  1471    10     0]
#  [   17   167   422    29     8  2830     0]
#  [  217    42     0     0     0     0  3843]]
# ------------------------------------
# Classification report:

#               precision    recall  f1-score   support

#            1       0.94      0.93      0.93     42368
#            2       0.94      0.95      0.94     56661
#            3       0.90      0.91      0.91      7151
#            4       0.84      0.73      0.78       549
#            5       0.85      0.77      0.81      1899
#            6       0.83      0.81      0.82      3473
#            7       0.95      0.94      0.94      4102

#     accuracy                           0.93    116203
#    macro avg       0.89      0.86      0.88    116203
# weighted avg       0.93      0.93      0.93    116203



# Epoch 1/10
# 2325/2325 [==============================] - 2s 711us/step - loss: 0.6235 - accuracy: 0.7374 - val_loss: 0.5302 - val_accuracy: 0.7701
# Epoch 2/10
# 2325/2325 [==============================] - 2s 658us/step - loss: 0.5047 - accuracy: 0.7832 - val_loss: 0.4812 - val_accuracy: 0.7951
# Epoch 3/10
# 2325/2325 [==============================] - 2s 657us/step - loss: 0.4693 - accuracy: 0.8004 - val_loss: 0.4586 - val_accuracy: 0.8069
# Epoch 4/10
# 2325/2325 [==============================] - 2s 655us/step - loss: 0.4468 - accuracy: 0.8117 - val_loss: 0.4342 - val_accuracy: 0.8190
# Epoch 5/10
# 2325/2325 [==============================] - 2s 658us/step - loss: 0.4299 - accuracy: 0.8195 - val_loss: 0.4220 - val_accuracy: 0.8226
# Epoch 6/10
# 2325/2325 [==============================] - 2s 677us/step - loss: 0.4167 - accuracy: 0.8257 - val_loss: 0.4104 - val_accuracy: 0.8289
# Epoch 7/10
# 2325/2325 [==============================] - 2s 658us/step - loss: 0.4064 - accuracy: 0.8306 - val_loss: 0.4137 - val_accuracy: 0.8267
# Epoch 8/10
# 2325/2325 [==============================] - 2s 656us/step - loss: 0.3969 - accuracy: 0.8352 - val_loss: 0.3983 - val_accuracy: 0.8328
# Epoch 9/10
# 2325/2325 [==============================] - 2s 655us/step - loss: 0.3893 - accuracy: 0.8384 - val_loss: 0.3927 - val_accuracy: 0.8381
# Epoch 10/10
# 2325/2325 [==============================] - 2s 657us/step - loss: 0.3833 - accuracy: 0.8414 - val_loss: 0.3816 - val_accuracy: 0.8423
# ===================================================
# Feed-forward Neural Network Performance Evaluation
# ===================================================
# Confusion matrix:

# [[33408  8613     9     0    34    14   290]
#  [ 3668 52080   437     0   128   309    39]
#  [    0   375  6295    44     3   434     0]
#  [    0     0   159   347     0    43     0]
#  [   93  1053    48     0   697     8     0]
#  [    4   527  1092    40     1  1809     0]
#  [  792    65     0     0     0     0  3245]]
# ---------------------------------------------------
# Classification report:

#               precision    recall  f1-score   support

#            1       0.88      0.79      0.83     42368
#            2       0.83      0.92      0.87     56661
#            3       0.78      0.88      0.83      7151
#            4       0.81      0.63      0.71       549
#            5       0.81      0.37      0.50      1899
#            6       0.69      0.52      0.59      3473
#            7       0.91      0.79      0.85      4102

#     accuracy                           0.84    116203
#    macro avg       0.82      0.70      0.74    116203
# weighted avg       0.84      0.84      0.84    116203



# ========================
# Comparison of accuracies
# ========================
# Approach                              Accuracy    CPU (secs)
# ----------------------------------  ----------  ------------
# Random forest                         0.954941    112.244
# Naive Bayes                           0.354999      0.162998
# K Nearest neighbours                  0.929735    915.68
# Feed-forward neural neural network    0.842328     16.7106
