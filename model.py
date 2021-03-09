# Importing the require libraries
import pandas as pd
import numpy as np
import pickle

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score

# Loading the data
data = pd.read_csv("transfusion.data")
# print(data.head())

# renaming the target column
data = data.rename(columns = {"whether he/she donated blood in March 2007":"target"})

# target incidence
print(data["target"].value_counts())
# Print target incidence proportions, rounding output to 3 decimal places
#data["target"].value_counts(normalize=True).round(3)

x = data.drop(columns = ["target"],axis = 1)
y = data["target"]

# Train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 0,stratify = y)

#AUTOML TPOT to find the best model 
tpot_cls = TPOTClassifier(generations = 10,
                           population_size = 100,
                           scoring = "roc_auc",
                           cv = 5,
                           n_jobs = -1,
                           verbosity = 2,
                          random_state = 0)
tpot_cls.fit(x_train,y_train)

# Print best pipeline steps
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot_cls.fitted_pipeline_.steps, start=1):
    # Print idx and transform
    print(f'{idx}. {transform}')
    
# y_pred_prob is a 2-D array of probability of being labeled as 0 (first  column of array) vs 1 (2nd column in array)
y_pred_prob = tpot_cls.predict_proba(x_test)

# AUC score for tpot model
tpot_roc_auc_score = roc_auc_score(y_test,y_pred_prob[:,1] )
print(f'\nAUC score for TPOT Best Model: {tpot_roc_auc_score:.4f}')

#Exporting the model
#tpot_cls.export("tpot_best_model.py")

# X_train's variance, rounding the output to 3 decimal places
print('\n',x_train.var().round(3))

#Lowering the variance of the feature with high variance using log transformation
x_train_normed,x_test_normed = x_train.copy(),x_test.copy()


# Log normalization
for data_set in [x_train_normed, x_test_normed]:
    # Add log normalized column
    data_set['Monetary (c.c. blood)'] = np.log(data_set["Monetary (c.c. blood)"])
   

#print('\n',x_train_normed.var().round(3),'\n')

# building logistic regression model
 
logit_reg_cls = LogisticRegression(random_state = 42, C = 25)
logit_reg_cls.fit(x_train_normed,y_train)

# AUC score for Logistic Regression model
logit_roc_auc_score = roc_auc_score(y_test, logit_reg_cls.predict_proba(x_test_normed)[:,1])
print(f'\nAUC score of logistic regression: {logit_roc_auc_score:.4f}')


#Saving the model
pickle.dump(logit_reg_cls,open('model.pkl','wb'))

#Loading the model to compare the result
#model = pickle.load(open('model.pkl','rb'))
#print("predection:", model.predict([[20,10,50,12]]))
