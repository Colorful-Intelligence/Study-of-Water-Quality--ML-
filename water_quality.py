"""
WATER QUALITY PROJECT
"""

#%% Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score
from collections import Counter


#%% Read the Dataset

data = pd.read_csv("water_potability.csv")

#%% EDA (Exploratory Data Analysis)

data.columns
"""
['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']
"""

data.info()

data.describe()

# In this dataset , Potability is a target column, I'm going to change the name of this column as a Target

data.rename({"Potability":"Target"},axis = 1,inplace = True)

#%% Outlier Detection

def detect_outliers(df,features):
    outlier_indices = []
    for c in features:
        # 1 st quartile
        Q1 = np.percentile(df[c],25)
        
        # 3 rd quartile
        Q3 = np.percentile(df[c],75)
        
        # IQR
        IQR = Q3 - Q1
        
        # Outlier step
        outlier_step = IQR * 1.5
   
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1-outlier_step) | (df[c] > Q3 + outlier_step)].index
        
        # store indeces
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers

data.loc[detect_outliers(data,['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity'])]


#%% Missing Values

data.columns[data.isnull().any()]
data.isnull().sum()  # ph = 491 , Sulfate = 781 , Trihalomethanes = 162

#%% Fill Missing Values

missing_values = ["ph","Sulfate","Trihalomethanes"]

for Z in missing_values:
    data[Z] = data[Z].fillna(data[Z].mean())

# To Check again for missing values
data.isnull().sum()


#%% Correlation Matrix (It shows us to understand relationship between features(columns))

corr_matrix = data.corr()
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(corr_matrix,annot = True,linewidths=0.5,fmt = ".0f",ax = ax)
plt.title("Correlation Between Features (Columns)")
plt.show()

#%% To get X and Y Coordinates

y = data.Target.values
x_data = data.drop(["Target"],axis = 1) # axis = 1 ,which means drop it as a column

#%% Normalization Operation

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

#%% Train-Test Split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#%% K-Nearst Neighors Classificiation

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)
predicted = knn.predict(x_test)

print("Accuracy of the KNN (k = 3) = % {}".format(accuracy_score(y_test,predicted)*100)) # Accuracy of the KNN (k = 3) = % 63.41463414634146

# Let's find best k value for KNN Algorithm

score_list = []

for each in range(1,100):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,100),score_list)
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K Value vs Accuracy")
plt.show()

# best k value is 26

knn3 = KNeighborsClassifier(n_neighbors = 26)
knn3.fit(x_train,y_train)
print("Accuracy of the KNN (k= 26): %{}".format(knn3.score(x_test,y_test)*100)) # Accuracy of the KNN (k= 26): %67.22560975609755

#%% Naive Bayes Classification

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
print("Accuracy of the Naive Bayes Classification : %{}".format(nb.score(x_test,y_test)*100)) # Accuracy of the Naive Bayes Classification : %63.109756097560975

#%% Random Forest Classification

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000,random_state=1) # n_estimators , which means number of trees
rf.fit(x_train,y_train)

rf_predicted = rf.predict(x_test)

print("Accuracy of the Random Forest Classification : % {}".format(accuracy_score(y_test,rf_predicted)*100))

"""
Accuracy of the Random Forest Classification : % 68.75
"""

#%% Confusion Matrix

y_pred = rf_predicted
x_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(x_true,y_pred)
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(cm,annot = True,linecolor = "red",fmt = ".0f")
plt.xlabel("y_true")
plt.ylabel("y_pred")
plt.title("Confusion Matrix")
plt.show()

#%% K-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuries = cross_val_score(estimator = rf, X = x_train ,y = y_train,cv = 10)

print("Average accuracy:% {} ".format(np.mean(accuries)*100))
print("Average std: % {} ".format(np.std(accuries)*100))

"""
Average accuracy:% 68.28244274809161 
Average std: % 1.540491303100044 
"""