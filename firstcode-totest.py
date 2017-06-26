'''
This code is used to do experiments on the training data, by splitting it into train-development sets.
Idea is to pick the best performing model-feature combination and use that to predict on test data.
caveat: test data is different from training data, as the prediction years are different (late 2015-2016). 
general note: We haven't used numpy before, still relatively newbies - One of us was more used to Java in the past, and the other - C/C++
'''

import numpy as np
import pandas as pd
from sklearn import linear_model,svm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


#Training and testing files
input_file = "train.csv"

#Read the csv file as a data frame object
df = pd.read_csv(input_file, header = 0)
print(len(df.columns))

#Add additional features - this turned out to be useless later.
df['year'], df['month'], df['day'] = df['timestamp'].str.split("-").str
df['year'] = df['year'].apply(pd.to_numeric)
df['month'] = df['month'].apply(pd.to_numeric)
print(len(df.columns))

#Gets the header for the file as a list
original_headers = list(df.columns.values)

#replace NaN values with 0
def removeNaNEtc(nparray):   
  nparray[np.isnan(nparray)] = 0
  return nparray

#Just checking if there are columns with infinite values - not being used anywhere now.
def checking_columns(numpy_array):
  print("Columns with infinite vals")
  for i in range(0,6):
   if not np.all(np.isfinite(numpy_array[:,i])):
    print(i)
 
  #checking if any column has NaN vals.
  print("Columns with NaN vals")
  for i in range(0,6):
   if not np.all(np.isnan(numpy_array[:,i])):
    print(i)


#Choosing only specific columns instead of everything. 
features_array = [] #2,3,4,5,6,7,8,9
features_array.extend(range(2,11))
features_array.extend(range(13,150))
#features_array.extend(range(153,291))
#features_array.extend([292,293])
#features_array.extend([13,15,16,19,22,23,25,31,32])
#features_array.extend(range(42,106))
#features_array.extend(range(108,114))
#features_array.extend(range(120,150))
features_array.append(291)
print(features_array)
#Note: This is just arbitrary choice. We did not do any feature analysis.

numpy_array = df.ix[:,features_array].as_matrix()
#print(numpy_array.shape)
#print("Total columns: ", numpy_array.shape[1])

#Remove all NAn values in feature array and replace them with zeroes
numpy_array = removeNaNEtc(numpy_array)

#splitting the data into features and prediction.
print(numpy_array.shape) 
all_data = numpy_array[:,0:-1]
all_preds = numpy_array[:,-1]

#Train-test split
train_data = numpy_array[:30000,0:-1]
test_data = numpy_array[30001:,0:-1]
train_preds = numpy_array[:30000,-1]
test_preds = numpy_array[30001:,-1]

#Exploring multiple models:
models = [linear_model.LinearRegression(), linear_model.Lasso(alpha = 0.1), linear_model.Lasso(alpha = 0.01), 
linear_model.Lasso(alpha = 10), linear_model.Ridge(alpha = 0.1), linear_model.Ridge(alpha = 0.01), linear_model.Ridge(alpha = 10), GradientBoostingRegressor(n_estimators=380), XGBRegressor()]
#, GradientBoostingRegressor(n_estimators=250, learning_rate=0.01, min_samples_split=5, max_depth=5)

#advice on GBR hyper parameter tuning: http://machinelearningmastery.com/configure-gradient-boosting-algorithm/

#these don't seem to work for this.
#linear_model.BayesianRidge(), linear_model.Lars(), svm.SVR()

for model in models:
   regr = model
   regr.fit(train_data,train_preds)
   model_preds = regr.predict(test_data)
   # learned coefficients
   #print('Coefficients: \n', regr.coef_)
   # MSE
   #print("Mean squared error: %.2f" % np.mean((regr.predict(test_data) - test_preds) ** 2))
   n = len(test_preds)
   model_preds[model_preds < 0] = 0
   print("****", model, "********")
   print("RMSLE: ", np.sqrt((1/n) * sum(np.square(np.log10(model_preds +1) - (np.log10(test_preds) +1)))))
   print("*************")
'''

#Just a few print statements to check stuff
#print(original_headers)
#print(len(original_headers))
#print(len(df))

#Just a few example print statements to access the data frame
#print(df['sport_count_5000'])
#print(df.columns.get_loc('price_doc'))
#print(len(df._get_numeric_data()))

#Converting the data frame to a numpy array.
#numpy_array = df.as_matrix()

'''
