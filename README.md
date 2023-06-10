
# Diabetes Prediction

# Aim:
     To develop a reliable algorithm that can accurately classify individuals as diabetic or non-diabetic based on their relevant features, aiding in early diagnosis and intervention for effective disease management.

# Algorithm:
Step 1: Import the necessary libraries, such as scikit-learn, pandas, and numpy.

Step 2: Load the dataset containing the relevant features and the target variable (diabetes status).

Step 3: Preprocess the data by handling missing values, if any, and performing any required data transformations (e.g., scaling).

Step 4: Split the dataset into training and testing sets, typically using an 80/20 or 70/30 split.

Step 5: Create an instance of the random forest classifier and set the desired parameters, such as the number of estimators and maximum depth.

Step 6: Train the random forest classifier using the training data.

Step 7: Predict the diabetes status for the testing data using the trained classifier.

Step 8: Evaluate the performance of the classifier by calculating relevant metrics such as accuracy, precision, recall, and F1 score.

# Program:
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('diabetes.csv')
df.head()
#lets describe the data
df.describe()
#infromation of dataset
df.info()
#any null values 
#not neccessary in above information we can see
df.isnull().values.any()
#histogram
df.hist(bins=10,figsize=(10,10))
plt.show()
#correlation
sns.heatmap(df.corr())
#lets count total outcome in each target 0 1
sns.countplot(y=df['Outcome'],palette='Set1')
sns.set(style="ticks")
sns.pairplot(df, hue="Outcome")
#box plot for outlier visualization
sns.set(style="whitegrid")
df.boxplot(figsize=(15,6))
#box plot
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(4,2)})
sns.boxplot(x=df['Insulin'])
plt.show()
sns.boxplot(x=df['BloodPressure'])
plt.show()
sns.boxplot(x=df['DiabetesPedigreeFunction'])
plt.show()
#outlier removal
Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR=Q3-Q1
print("---Q1--- \n",Q1)
print("\n---Q3--- \n",Q3)
print("\n---IQR---\n",IQR)
#print((df < (Q1 - 1.5 * IQR))|(df > (Q3 + 1.5 * IQR)))
df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape,df_out.shape
#Scatter matrix after removing outlier
sns.set(style="ticks")
sns.pairplot(df_out, hue="Outcome")
plt.show()
#lets extract features and targets
X=df_out.drop(columns=['Outcome'])
y=df_out['Outcome']
#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
acc=[]
roc=[]
clf=LogisticRegression()
clf.fit(train_X,train_y)
y_pred=clf.predict(test_X)
#find accuracy
ac=accuracy_score(test_y,y_pred)
acc.append(ac)
#find the ROC_AOC curve
rc=roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))
#cross val score
result=cross_validate(clf,train_X,train_y,scoring=scoring,cv=10)
display_result(result)
#display predicted values uncomment below line
pd.DataFrame(data={'Actual':test_y,'Predicted':y_pred}).head()
#KNN
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(train_X,train_y)
y_pred=clf.predict(test_X)
#find accuracy
ac=accuracy_score(test_y,y_pred)
acc.append(ac)
#find the ROC_AOC curve
rc=roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))
#cross val score
result=cross_validate(clf,train_X,train_y,scoring=scoring,cv=10)
display_result(result)
#display predicted values uncomment below line
#pd.DataFrame(data={'Actual':test_y,'Predicted':y_pred}).head()
#Random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(train_X,train_y)
y_pred=clf.predict(test_X)
#find accuracy
ac=accuracy_score(test_y,y_pred)
acc.append(ac)
#find the ROC_AOC curve
rc=roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))
#cross val score
result=cross_validate(clf,train_X,train_y,scoring=scoring,cv=10)
display_result(result)
pd.DataFrame(data={'Actual':test_y,'Predicted':y_pred}).head()
#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier(n_estimators=50,learning_rate=0.2)
clf.fit(train_X,train_y)
y_pred=clf.predict(test_X)
#find accuracy
ac=accuracy_score(test_y,y_pred)
acc.append(ac)
#find the ROC_AOC curve
rc=roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))
#cross val score
result=cross_validate(clf,train_X,train_y,scoring=scoring,cv=10)
display_result(result)
#display predicted values uncomment below line
#pd.DataFrame(data={'Actual':test_y,'Predicted':y_pred}).head()
#lets plot the bar graph
ax=plt.figure(figsize=(9,4))
plt.bar(['Logistic Regression','SVM','KNN','Random Forest','Naivye Bayes','Gradient Boosting'],acc,label='Accuracy')
plt.ylabel('Accuracy Score')
plt.xlabel('Algortihms')
plt.show()
ax=plt.figure(figsize=(9,4))
plt.bar(['Logistic Regression','SVM','KNN','Random Forest','Naivye Bayes','Gradient Boosting'],roc,label='ROC AUC')
plt.ylabel('ROC AUC')
plt.xlabel('Algortihms')
plt.show()
```
# Output:
![]("C:\Users\harih\Pictures\DS\download.png")

![]("C:\Users\harih\Pictures\DS\download (1).png")

![]("C:\Users\harih\Pictures\DS\download (2).png")

![]("C:\Users\harih\Pictures\DS\download (3).png")

# Result:
 Thus, the result of diabetes prediction for the given dataset was found using random forest classsifier indicating whether the individual is predicted to have diabetes or not.

