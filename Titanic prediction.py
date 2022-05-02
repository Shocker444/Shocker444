import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv('C:/Users/SHOCKER/Downloads/Dataset for jupyter/Titanic Dataset/train.csv', low_memory=False)
eval = pd.read_csv('C:/Users/SHOCKER/Downloads/Dataset for jupyter/Titanic Dataset/eval.csv', low_memory=False)

print(data.head())

# Checking for missing values
print(data.isnull().sum())

le = LabelEncoder()


def preprocessing(Data):
    # converting to number using label encoder
    Data['sex'] = le.fit_transform(Data['sex'])
    Data['class'] = le.fit_transform(Data['class'])
    Data['alone'] = le.fit_transform(Data['alone'])

    # converting to number using one hot encoding
    categories = ['embark_town', 'deck']
    Data = pd.get_dummies(Data, prefix_sep='-', columns=categories)

    return Data


Data = preprocessing(data)
print(Data.head())

y1 = Data['survived']
X1 = Data.drop(['survived'], axis=1)

y = np.array(y1)
X = np.array(X1)

scalar = StandardScaler()
X = scalar.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X_train.shape)
print(y_train.shape)


def estimator(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    print('Accuracy of the model on test data is: ', accuracy)


estimator(LogisticRegression(), X_train, y_train, X_test, y_test)
estimator(ExtraTreesClassifier(), X_train, y_train, X_test, y_test)
estimator(xgb.XGBClassifier(use_label_encoder=False, eval_metric='error'), X_train, y_train, X_test, y_test)
estimator(RandomForestClassifier(), X_train, y_train, X_test, y_test)
estimator(KNeighborsClassifier(), X_train, y_train, X_test, y_test)

model = ExtraTreesClassifier()
model.fit(X1, y1)
feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()