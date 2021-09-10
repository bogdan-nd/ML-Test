# ML-Test
## Bogdan Naida
ML Test Assignment.
* EDA, my implementation of logistic regression, and SKLearn Classification models for Heart Disease UCI
* EDA, my implementation of linear regression, SKLearn Regression models for Medical Cost Personal Dataset


### 1. For which tasks is it better to use Logistic Regression instead of other models?
For binary classification

### 2. What are the most important parameters that you can tune in Linear Regression / Logistic Regression / SVM?
For Linear Regression ( GD optimizer):
* Learning Rate - step size at each iteration while moving toward the minimum of a loss function
* Regularization Term - reduces overfitting

For Logistic Regression ( GD optimizer):
* Learning Rate - step size at each iteration while moving toward the minimum of a loss function
* Regularization Term - reduces overfitting
* C - regularization parameter

For SVM ( GD optimizer):
* C - regularization parameter
* kernel - function, that takes data as input and transform it into the required form

### 3. How does parameter C influence regularisation in Logistic Regression??
The strength of the regularization is inversely proportional to C.<br>
Higher C - lower regularization and vice versa.<br>

### 4. Which top 3 features are the most important for each data sets?
For Heart Disease UCI:
1. ca (number of major vessels (0-3) colored by fluoroscopy)
2. oldpeak (ST depression induced by exercise relative to rest) 
3. exang (exercise induced angina)

Medical Cost Personal Dataset:
1. smoker
2. age
3. BMI

### 5. What accuracy metrics did you receive on train and test sets for Heart Disease UCI dataset?
#### Train:
Accuracy: 0.86<br>
F-1: 0.86

#### Test:
Accuracy: 0.85<br>
F-1: 0.84

### 6. What MSE did you receive on train and test datasets for Medical Cost Personal?
#### Train:
MSE: 9161099

#### Test:
MSE: 36099509

### 7. Try a few different models and compare them.
For the classification problem I've tried:
* SKLearn Logistic Regression
* Random Forest Classifier
* MLPClassifier

The best result I had was with the SKLearn Logistic regression model, which was identical to my Logistic Regression implementation.
The Random Forest Classifier even with 300 estimators hasn't shown better metrics, than my simple regression.

For the regression problem I've tried:
* SKLearn Linear Regression
* Random Forest Regressor
* MLPRegressor

The best result I had was with the Random Forest Classifier regression model, mse = 24433024.
Also, I have great results with SKLearn Linear Regression, that have quite similar mse.
