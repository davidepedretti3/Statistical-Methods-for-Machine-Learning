# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

# this is useful for reproducibility 
np.random.seed(0)
tf.random.set_seed(0)

# loading of training data
data = genfromtxt('features_for_training.csv', delimiter=',')   

# loading of testing data
test5 = genfromtxt('test5.csv', delimiter=',')  
test7 = genfromtxt('test7.csv', delimiter=',')  
test8 = genfromtxt('test8.csv', delimiter=',')  
test9= genfromtxt('test9.csv', delimiter=',')  
test10= genfromtxt('test10.csv', delimiter=',')  

data_for_test = [test5, test7, test8, test9, test10]

df=pd.read_csv(r"UrbanSound8K.csv")
dataclass=np.unique(df["class"])
mapping=dataclass.tolist() 

labels = data[:,0]
training_data = data[:,1:]

# standardizing the features 
standardScaler = StandardScaler()
scaled_features = standardScaler.fit_transform(training_data) 
'''
# pca for dimensionality reduction
pca = PCA(0.99)
pca.fit(scaled_features)
scaled_features = pca.transform(scaled_features)
'''

# splitting the data using the hold out technique 
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.3, random_state=15)

def get_model(first_hlayer_size, second_hlayer_size, third_hlayer_size, learn_rate):
    # Create Model and return it 
    mlp = keras.Sequential([
        # input layer and first dense layer
        keras.layers.Dense(first_hlayer_size,input_shape=(X_train.shape[1],), activation='relu'),
        # second dense layer
        keras.layers.Dense(second_hlayer_size, activation='relu'),
        keras.layers.Dropout(0.4),
        # third dense layer
        keras.layers.Dense(third_hlayer_size, activation='relu'),
        keras.layers.Dropout(0.4),
        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile Model
    mlp.compile(optimizer=keras.optimizers.Adam(learning_rate=learn_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return mlp

model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=get_model,verbose=0)

# define the grid search parameters
first_hlayer_size = [96, 128]
second_hlayer_size = [48, 64]
third_hlayer_size = [16, 32]
batch_size = [32, 64]
epochs = [40, 70, 100]
learn_rate = [0.01, 0.001]
hparams = dict(first_hlayer_size = first_hlayer_size,
               second_hlayer_size = second_hlayer_size,
               third_hlayer_size = third_hlayer_size,
               batch_size=batch_size,
               epochs=epochs,
               learn_rate=learn_rate)

'''
grid = GridSearchCV(estimator=model, param_grid=hparams, scoring='accuracy', n_jobs=1,cv=3)
grid_result = grid.fit(X_train, y_train)
score = grid_result.best_score_
final_parameters = grid_result.best_params_
print("Using the Grid Search, the best score is {:.2f} using {}".format(score,
	final_parameters))
'''
random_search = RandomizedSearchCV(estimator=model, n_jobs=1, cv=3, n_iter=10,
	param_distributions=hparams, scoring="accuracy")
random_search_results = random_search.fit(X_train, y_train)
# summarize grid search information
score = random_search_results.best_score_
final_parameters = random_search_results.best_params_
print("Using the Randomized Search, the best score is {:.2f} using {}".format(score,
	final_parameters))

model_def = get_model(final_parameters['first_hlayer_size'],final_parameters['second_hlayer_size'],final_parameters['third_hlayer_size'],final_parameters['learn_rate'])

# it returns a summary of the network
model_def.summary() 

# it trains the model for a fixed number of epochs
history = model_def.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=final_parameters['batch_size'], epochs=final_parameters['epochs'])

# testing part 

accuracies = []
for fold in data_for_test:    
    labels_test = fold[:,0]
    testing_data = fold[:,1:]
    scaled_test = standardScaler.fit_transform(testing_data) 
    
    # scaled_test = pca.transform(scaled_test)
    
    # it returns the loss value & metrics values for the model in test mode.
    model_def.evaluate(scaled_test, labels_test, verbose=0)
    
    # this part's aim is to obtain a confusion matrix
    predictions = model_def.predict(scaled_test)
    
    predicted_labels=[]
    for prediction in predictions:
        max_value = prediction[0]
        label = 0
        for x in range(len(prediction)):
            if prediction[x] > max_value:
                max_value = prediction[x]
                label = x
        predicted_labels.append(label)
    
    matrix = confusion_matrix(labels_test, predicted_labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=mapping)
    disp.plot(xticks_rotation='vertical')
    plt.show()
    
    # calculate accuracy
    accuracy = accuracy_score(labels_test, predicted_labels)
    accuracies.append(accuracy)

print("The avarage accuracy across the test folders is: ", np.average(accuracies))
print("The standard deviation across the test folders is: ", np.std(accuracies))
