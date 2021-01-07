import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Read the CSV file into a DataFrame: df
gapminder=pd.read_csv('Gapminder.csv')
print(gapminder.head())
print(gapminder.info())
print(gapminder.describe())
# Create arrays for features and target variable

#axis=1, colunms=labels, axis=0, index=labels

y=gapminder['GDP'].values #only the target value


X=gapminder['fertility'].values

#Print the dimemsions of X and y before reshaping
print("Dimentions of y before reshaping:{}".format(y.shape))
print("Dimentions of X before reshaping:{}".format(X.shape))

#Reshapte X and y
y=y.reshape(-1,1)#keep the first dimention (y)==-1, and add another dimention 1 to x==1
X=X.reshape(-1,1)

#Print the dimemsions of X and y after reshaping
print("Dimentions of y before reshaping:{}".format(y.shape))
print("Dimentions of X before reshaping:{}".format(X.shape))

# create heatmap
sns.heatmap(gapminder.corr(), cmap="RdYlBu")

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space

prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)

plt.show()
# Fit the model to the data
reg.fit(X,y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)



# Plot regression line
plt.plot(X,y,'o')
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.ylabel(' GDP ($)')
plt.xlabel('fertility')
plt.show()

# Print R^2
print(reg.score(X,y))

sns.regplot(x=X,y=y,color="g",marker="*")

# Create training and test sets

y=gapminder['GDP'] #drop the target
X=gapminder[['population','fertility','HIV','CO2','BMI_female','life','child_mortality']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=5)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred)) # compare with the real value with the predicted values
print("Root Mean Squared Error: {}".format(rmse))

#Crossvaluation with 5 folds

cv_scores=cross_val_score(reg_all,X,y,cv=5)
print(cv_scores)
print("Average 5-fold CV score:{}".format(np.mean(cv_scores)))


#Regularized regression to avoid the overfitting



# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4,normalize=True)

# Fit the regressor to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.fit(X,y).coef_
print(lasso_coef)

## Plot the coefficients
df_columns=X.columns
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.ylabel('coefficients')
plt.xlabel('features')
plt.margins(0.02)
plt.show()

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge,X,y,cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

# Display the plot
display_plot(ridge_scores, ridge_scores_std)


#how good the model is

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=45)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
