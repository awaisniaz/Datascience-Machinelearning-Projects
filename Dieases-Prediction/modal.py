import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))
if __name__=="__main__":
    traind_data = pd.DataFrame(pd.read_csv('Training.csv'))
    testinf_data = pd.read_csv('Testing.csv')
    traind_data = traind_data.drop('Unnamed: 133',axis=1)
    print(traind_data.head())
    disease_counts = traind_data['prognosis'].value_counts()
    encoder = LabelEncoder()
    traind_data['prognosis'] = encoder.fit_transform(traind_data['prognosis'])
    print(traind_data['prognosis'])
    X = traind_data.drop('prognosis',axis =1)
    Y = traind_data['prognosis']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    models = {
        "SVC": SVC(),
        "Gaussian NB": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=18)
    }

    # Producing cross validation score for the models
    for model_name in models:
        model = models[model_name]
        scores = cross_val_score(model, X, y, cv=10,
                                 n_jobs=-1,
                                 scoring=cv_scoring)
        print("==" * 30)
        print(model_name)
        print(f"Scores: {scores}")
        print(f"Mean Score: {np.mean(scores)}")

