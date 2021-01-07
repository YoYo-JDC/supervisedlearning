#import all needed package and rename them

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#load the data and create the keys for the dataframe
vote1=pd.read_csv('vote84.data',header=None,names=(['party','infants','water','budget','physician','slvador','religious','satellite','aid','missile','immigration','synfuels','edu','superfund','crime','duty_free','eaa_rsa']))
#elmi
vote=vote1.replace(['n','?','y'],[0,0,1])
print(type(vote))
print(vote.keys())
print(vote.shape)
print(vote.shape)
print(vote.head())
print(vote.info())
print(vote.describe())



plt.figure()
sns.countplot(x='infants',hue='party',data=vote,palette='RdBu')
plt.xticks([0,1],['No','Yes'])

#plt.figure()
#sns.countplot(x='water',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='budget',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='physician',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='slvador',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='religious',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='satellite',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='aid',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='missile',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='immigration',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='synfuels',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='edu',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='superfund',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='crime',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='duty_free',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])


#plt.figure()
#sns.countplot(x='eaa_rsa',hue='party',data=vote,palette='RdBu')
#plt.xticks([0,1],['No','Yes'])
plt.show()

# Create arrays for the features and the response variable
y = vote['party'].values
X = vote.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)

#define X_new

X_new=[[0.696469, 0.286139, 0.226851, 0.551315, 0.719469, 0.423106, 0.980764, 0.68483, 0.480932, 0.392118, 0.343178, 0.72905, 0.438572, 0.059678, 0.398044, 0.737995]]
print(len(X_new))
# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)
#test_size: the 30% of the date will be used to do as test set (Defualt 75% for trainging set and 25% for testing set);
#random_state:Controls the shuffling applied to the data before applying the split.
#stratify:If not None, data is split in a stratified fashion, using this as the class labels

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=8)

# Fit the classifier to the data
knn.fit(X_train,y_train)

test_prediction = knn.predict(X_test)
print("Test set Prediction: {}".format(test_prediction))

#test the quality of the Prediction
print(knn.score(X_test, y_test))



# compute and plot the training and testing accuracy scores for a variety of different neighbor values.
#By observing how the accuracy scores differ for the training and testing sets with different values of k,
#you will develop your intuition for overfitting and underfitting.

#The training and testing sets are available to you in the workspace as X_train, X_test, y_train, y_test.
#In addition, KNeighborsClassifier has been imported from sklearn.neighbors.
# Setup arrays to store train and test accuracies

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
